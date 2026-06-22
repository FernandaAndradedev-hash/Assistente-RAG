"""
API HTTP do assistente RAG.

Endpoints:
  GET  /health       — verifica saúde do serviço (usado por load balancers)
  POST /ask          — faz uma pergunta ao assistente
  POST /ingest       — ingere um arquivo de documento

Princípios aplicados:
- Toda entrada é validada antes de chegar à lógica de negócio
- Erros do usuário retornam 4xx; erros internos retornam 5xx
- Nunca expor detalhes de erros internos ao usuário
- Logging estruturado para facilitar debugging em produção
"""
import logging
import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chain import ask
from ingest import ingest_file
from validators import sanitize_query


# Logging ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# Aplicação FastAPI ─────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Assistant API",
    description="Chatbot sobre documentos internos com RAG seguro.",
    version="1.0.0",
    # docs_url="/docs" em desenvolvimento.
    # Em produção, mude para docs_url=None para não expor a documentação.
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: define quais origens podem chamar a API
# Em produção, substitua "*" pelo domínio real do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",  # Chainlit em desenvolvimento
        # "https://seu-dominio.com",  # adicione em produção
    ],
    allow_credentials=False,       # não usamos cookies
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Modelos de request/response (Pydantic valida automaticamente)─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Pergunta para o assistente.",
        examples=["Qual é o prazo de entrega descrito no contrato?"],
    )


class AskResponse(BaseModel):
    answer: str = Field(description="Resposta gerada pelo assistente.")
    sources: list[str] = Field(description="Arquivos usados como fonte.")
    chunks_used: int = Field(description="Número de trechos utilizados como contexto.")
    no_context: bool = Field(description="True se nenhum trecho relevante foi encontrado.")


class IngestResponse(BaseModel):
    file: str
    hash: str
    chunks_ingested: int
    skipped: bool


class HealthResponse(BaseModel):
    status: str
    version: str


# Endpoints ───────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Infraestrutura"],
    summary="Verifica saúde do serviço",
)
async def health():
    """Endpoint de health check. Retorna 200 se o serviço está rodando."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["Assistente"],
    summary="Faz uma pergunta ao assistente",
    status_code=status.HTTP_200_OK,
)
async def ask_endpoint(body: AskRequest):
    """
    Processa uma pergunta e retorna a resposta baseada nos documentos indexados.

    - Sanitiza a entrada contra prompt injection e HTML
    - Busca chunks relevantes no Qdrant
    - Gera resposta com Claude usando apenas o contexto recuperado
    """
    # Sanitização (segunda camada — Pydantic já validou tamanho mínimo/máximo)
    try:
        clean_question = sanitize_query(body.question)
    except ValueError as exc:
        # 422 Unprocessable Entity — erro do usuário, não do servidor
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    try:
        result = ask(clean_question)
        return AskResponse(**result)
    except Exception as exc:
        # Loga o erro completo internamente
        logger.error("Erro ao processar pergunta: %s", exc, exc_info=True)
        # Retorna mensagem genérica ao usuário — não expõe detalhes internos
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao processar sua pergunta. Tente novamente em instantes.",
        )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Documentos"],
    summary="Ingere um arquivo de documento",
    status_code=status.HTTP_201_CREATED,
)
async def ingest_endpoint(file: UploadFile = File(...)):
    """
    Recebe um arquivo e o indexa no banco vetorial.

    Tipos aceitos: PDF, TXT, MD
    Tamanho máximo: configurável via MAX_FILE_SIZE_BYTES no .env
    """
    # Salva o arquivo temporariamente em docs/
    dest = Path("docs") / file.filename

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = ingest_file(str(dest))
        return IngestResponse(**result)

    except (ValueError, FileNotFoundError) as exc:
        # Remove o arquivo temporário se a validação falhou
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.error("Erro na ingestão de '%s': %s", file.filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao processar o arquivo. Verifique se não está corrompido.",
        )
    finally:
        # Garante que o arquivo de upload é fechado
        await file.close()

 
# Execução direta ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,    # reinicia ao salvar arquivos (só em desenvolvimento)
        log_level="info",
    )