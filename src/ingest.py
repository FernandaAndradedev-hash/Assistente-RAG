"""
Pipeline de ingestão de documentos.

Responsabilidade única: pegar um arquivo do disco e indexá-lo no Qdrant.

Fluxo completo:
  arquivo → validação → extração de texto → chunking → embedding → Qdrant

Por que este módulo não conhece o FastAPI nem o Chainlit?
  Separação de responsabilidades. A ingestão pode ser chamada via CLI,
  via API, via script de cron — sem mudar uma linha daqui.
"""
import logging
import uuid
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config
from validators import compute_file_hash, sanitize_chunk_text, validate_file

logger = logging.getLogger(__name__)



#
#Clientes (inicializados uma vez, reutilizados em todas as chamadas)─────────────────────────────────────────────────────────────────────────────

# Por que inicializar no nível de módulo e não dentro das funções?
# Criar um cliente a cada chamada de função geraria overhead de conexão desnecessário. Estes objetos são thread-safe.

_openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
_qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)



# Funções internas (prefixo _ = não fazem parte da API pública do módulo)─────────────────────────────────────────────────────────────────────────────

def _ensure_collection() -> None:
    """
    Garante que a coleção existe no Qdrant. Cria se não existir.

    Esta função é idempotente: pode ser chamada múltiplas vezes sem efeito
    negativo. Na prática, só cria a coleção na primeira ingestão.

    Por que não criar a coleção no docker-compose?
    O Qdrant não tem suporte a inicialização com coleções predefinidas via
    variável de ambiente. A criação programática é a abordagem padrão.
    """
    existing_names = {col.name for col in _qdrant_client.get_collections().collections}

    if config.QDRANT_COLLECTION not in existing_names:
        _qdrant_client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIMENSIONS,   # 1536 para text-embedding-3-small
                distance=Distance.COSINE,            # padrão para embeddings de texto
            ),
        )
        logger.info("Coleção '%s' criada no Qdrant.", config.QDRANT_COLLECTION)
    else:
        logger.debug("Coleção '%s' já existe.", config.QDRANT_COLLECTION)


def _load_document(path: Path) -> list[str]:
    """
    Extrai texto bruto de um arquivo.

    Retorna uma lista de strings. Para PDFs, cada elemento é o texto
    de uma página. Para TXT/MD, é uma lista com um único elemento.

    Por que separar por página em PDFs?
    Permite armazenar o número da página no payload do Qdrant, o que
    viabiliza citações precisas na resposta ("conforme página 12 do contrato").

    Args:
        path: Caminho validado do arquivo.

    Returns:
        Lista de strings com o texto extraído.

    Raises:
        ValueError: Se o tipo de arquivo não tiver loader disponível.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # PyPDFLoader carrega cada página como um Document separado
        loader = PyPDFLoader(str(path))
        pages = loader.load()

        if not pages:
            raise ValueError(f"Nenhuma página extraída de '{path.name}'. PDF pode ser escaneado sem OCR.")

        return [page.page_content for page in pages]

    if suffix in {".txt", ".md"}:
        # TextLoader carrega o arquivo inteiro como um único Document
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        return [doc.page_content for doc in docs]

    # Não deve chegar aqui se validate_file foi chamado antes
    raise ValueError(f"Nenhum loader disponível para extensão '{suffix}'")


def _chunk_texts(texts: list[str]) -> list[str]:
    """
    Divide uma lista de textos em chunks menores.

    Usa RecursiveCharacterTextSplitter, que tenta manter unidades
    semânticas intactas dividindo na seguinte ordem de preferência:
      \\n\\n (parágrafo) → \\n (linha) → ". " (frase) → " " (palavra) → "" (caractere)

    Cada texto passa pela sanitização antes do chunking para garantir
    que caracteres problemáticos não entrem no banco.

    Args:
        texts: Lista de strings extraídas do documento.

    Returns:
        Lista de chunks prontos para embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,       # 800 caracteres ≈ 200 palavras
        chunk_overlap=config.CHUNK_OVERLAP, # 150 chars de sobreposição entre chunks
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[str] = []
    for text in texts:
        # Sanitiza antes de chunkar — remove chars de controle, etc.
        clean_text = sanitize_chunk_text(text)

        if not clean_text:
            # Página vazia ou apenas espaços — ignorar
            continue

        chunks = splitter.split_text(clean_text)
        all_chunks.extend(chunks)

    return all_chunks


def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Gera embeddings para uma lista de chunks.

    Envia em lotes (batches) para não exceder os limites da API da OpenAI
    e para reduzir o número de requisições HTTP.

    A API da OpenAI aceita até 2048 inputs por chamada, mas mantemos
    lotes de 100 para segurança e para não travar em caso de erro.

    Args:
        chunks: Lista de strings para gerar embeddings.

    Returns:
        Lista de vetores (listas de floats), na mesma ordem dos chunks.

    Raises:
        openai.APIError: Em caso de falha na API (automaticamente logado).
    """
    BATCH_SIZE = 100
    all_embeddings: list[list[float]] = []

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, len(chunks), BATCH_SIZE), 1):
        batch = chunks[i : i + BATCH_SIZE]

        response = _openai_client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
        )

        # A API retorna os embeddings na mesma ordem dos inputs
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        logger.debug(
            "Embedding: lote %d/%d (%d chunks processados)",
            batch_num,
            total_batches,
            min(i + BATCH_SIZE, len(chunks)),
        )

    return all_embeddings


def _file_already_ingested(file_hash: str) -> bool:
    """
    Verifica se um arquivo com este hash já foi indexado.

    Usa o payload do Qdrant em vez de uma busca vetorial —
    mais rápido e direto para esta finalidade.

    Args:
        file_hash: SHA-256 do arquivo.

    Returns:
        True se o arquivo já foi ingerido anteriormente.
    """
    results, _ = _qdrant_client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter={
            "must": [
                {"key": "file_hash", "match": {"value": file_hash}}
            ]
        },
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(results) > 0


# API pública do módulo ─────────────────────────────────────────────────────

def ingest_file(file_path: str | Path) -> dict:
    """
    Ingere um único arquivo no Qdrant.

    Este é o ponto de entrada principal do módulo. Orquestra todas as
    etapas do pipeline.

    Args:
        file_path: Caminho para o arquivo a ser ingerido.

    Returns:
        Dicionário com resultado da operação:
        {
            "file": str,           # nome do arquivo
            "hash": str,           # SHA-256 do arquivo
            "chunks_ingested": int, # número de chunks indexados
            "skipped": bool,       # True se o arquivo já havia sido ingerido
        }

    Raises:
        ValueError: Para arquivos inválidos.
        FileNotFoundError: Se o arquivo não existir.
    """
    _ensure_collection()

    # Etapa 1: Validação do arquivo
    path = validate_file(file_path)
    file_hash = compute_file_hash(path)

    logger.info("Iniciando ingestão: '%s' (hash: %s...)", path.name, file_hash[:8])

    # Etapa 2: Verificar duplicata
    if _file_already_ingested(file_hash):
        logger.info("Arquivo '%s' já ingerido anteriormente. Pulando.", path.name)
        return {
            "file": path.name,
            "hash": file_hash,
            "chunks_ingested": 0,
            "skipped": True,
        }

    # Etapa 3: Extrair texto
    texts = _load_document(path)
    logger.info("'%s': %d páginas/seções extraídas.", path.name, len(texts))

    # Etapa 4: Chunking
    chunks = _chunk_texts(texts)

    if not chunks:
        raise ValueError(
            f"Nenhum chunk gerado de '{path.name}'. "
            "O arquivo pode estar corrompido, ser um PDF escaneado sem OCR, "
            "ou conter apenas imagens."
        )

    logger.info("'%s': %d chunks gerados.", path.name, len(chunks))

    # Etapa 5: Gerar embeddings
    embeddings = _embed_chunks(chunks)

    # Etapa 6: Montar e inserir pontos no Qdrant
    # Cada ponto tem: ID único, vetor, e payload com metadados
    points = [
        PointStruct(
            id=str(uuid.uuid4()),  # UUID v4 garante unicidade sem estado global
            vector=embedding,
            payload={
                "text": chunk,           # texto original do chunk (para exibir na resposta)
                "source": path.name,     # nome do arquivo (para citação)
                "file_hash": file_hash,  # para deduplicação e auditoria
                "chunk_index": idx,      # posição no documento (para ordenação futura)
            },
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    _qdrant_client.upsert(
        collection_name=config.QDRANT_COLLECTION,
        points=points,
        wait=True,  # aguarda confirmação de escrita antes de retornar
    )

    logger.info(
        "✅ Ingestão concluída: '%s' → %d chunks indexados.",
        path.name,
        len(points),
    )

    return {
        "file": path.name,
        "hash": file_hash,
        "chunks_ingested": len(points),
        "skipped": False,
    }


def ingest_directory(docs_dir: str = "docs") -> list[dict]:
    """
    Ingere todos os arquivos suportados em um diretório.

    Processa arquivos de forma sequencial e continua mesmo se um arquivo
    falhar — outros arquivos não são prejudicados por erros individuais.

    Args:
        docs_dir: Caminho para o diretório de documentos.

    Returns:
        Lista de resultados, um por arquivo processado.
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Diretório '{docs_dir}' não encontrado.")

    results: list[dict] = []

    for ext in config.ALLOWED_EXTENSIONS:
        for file_path in sorted(docs_path.rglob(f"*{ext}")):
            try:
                result = ingest_file(file_path)
                results.append(result)
            except Exception as exc:
                logger.error("Erro ao ingerir '%s': %s", file_path.name, exc)
                results.append({
                    "file": file_path.name,
                    "error": str(exc),
                    "skipped": False,
                    "chunks_ingested": 0,
                })

    total_chunks = sum(r.get("chunks_ingested", 0) for r in results)
    logger.info(
        "Ingestão do diretório concluída: %d arquivos, %d chunks totais.",
        len(results),
        total_chunks,
    )

    return results


# Permite rodar diretamente: python src/ingest.py
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = ingest_directory()
    print(json.dumps(results, indent=2, ensure_ascii=False))