"""
Camada de segurança central do projeto.

Princípio: validar e sanitizar TODA entrada externa antes de processá-la.
Entradas externas: perguntas do usuário, arquivos enviados, parâmetros de API.

Este módulo não depende de nenhum outro módulo do projeto (exceto config).
O tornando testável de forma completamente isolada.
"""
import hashlib
import logging
import re
from pathlib import Path

import bleach

from config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_QUERY_LENGTH,
)

logger = logging.getLogger(__name__)


#  Prompt Injection — padrões de detecção ──────────────────────────────────────

# Esta lista cobre ataques documentados publicamente.
# Referências:
#   - https://github.com/greshake/llm-security
#   - https://arxiv.org/abs/2302.12173 (Perez & Ribeiro, 2022)
#   - OWASP LLM Top 10: LLM01 Prompt Injection

# Por que regex e não LLM para detectar injection?
# Usar um LLM para detectar injection cria dependência circular e custo.
# Regex é determinístico, gratuito e suficiente para os padrões conhecidos.
# A segunda camada (system prompt do Claude) cobre casos que escapam do regex.

_INJECTION_PATTERNS: list[str] = [
    # Comandos de ignorar instruções 
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"override\s+(all\s+)?(previous|prior)\s+instructions?",

    # Comandos de esquecer
    r"forget\s+(everything|all|your|the\s+previous)",
    r"clear\s+(your\s+)?(memory|context|instructions?)",

    # Mudança de identidade
    r"you\s+are\s+now\s+(a|an)\s+\w+",
    r"act\s+as\s+(if\s+you\s+are|a|an)\s+",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"roleplay\s+as\s+",
    r"simulate\s+(being\s+)?(a|an)\s+",

    # Injeção de novos prompts
    r"new\s+(system\s+)?instructions?\s*:",
    r"updated?\s+instructions?\s*:",
    r"system\s+prompt\s*:",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"\[SYSTEM\]",
    r"###\s*instruction",
    r"```\s*system",

    # Jailbreaks conhecidos
    r"do\s+anything\s+now",
    r"developer\s+mode",
    r"jailbreak",
    r"\bdan\b.*\bmode\b",   # "DAN mode"
    r"unrestricted\s+mode",
    r"no\s+restrictions?",

    # Pedidos de revelar o system prompt
    r"(show|print|repeat|reveal|tell\s+me)\s+(me\s+)?(your\s+)?(system\s+prompt|instructions?|rules?)",
    r"what\s+(are\s+)?(your\s+)?(instructions?|rules?|guidelines?|constraints?)",
]

# Compilado uma vez na importação do módulo para eficiência
_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)


# Funções públicas ────────────────────────────────────────────────────────────

def sanitize_query(text: str) -> str:
    """
    Limpa e valida a pergunta do usuário.

    Etapas (nesta seguinte ordem):
    1. Verifica tipo (string)
    2. Remove HTML e JavaScript via bleach
    3. Normaliza espaços
    4. Verifica que não está vazia
    5. Verifica tamanho máximo
    6. Detecta padrões de prompt injection

    Args:
        text: Texto bruto enviado pelo usuário.

    Returns:
        Texto limpo e seguro para processar.

    Raises:
        TypeError: Se `text` não for string.
        ValueError: Se a entrada for inválida ou suspeita.
    """
    if not isinstance(text, str):
        raise TypeError(f"Esperado str, recebido {type(text).__name__}")

    # bleach.clean() com tags=[] e strip=True remove TODAS as tags HTML.
    # Mais confiável que regex para HTML porque lida com variações como
    # <SCRIPT>, <scri\x00pt>, &#60;script&#62;, etc.
    clean = bleach.clean(text, tags=[], strip=True)

    # Normaliza múltiplos espaços/tabs/newlines em espaço único
    clean = re.sub(r"\s+", " ", clean).strip()

    if not clean:
        raise ValueError("A pergunta não pode estar vazia.")

    if len(clean) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"Pergunta muito longa ({len(clean)} caracteres). "
            f"Limite: {MAX_QUERY_LENGTH} caracteres."
        )

    match = _INJECTION_RE.search(clean)
    if match:
        # Log com contexto suficiente para auditoria, sem expor dados sensíveis completos
        logger.warning(
            "Tentativa de prompt injection bloqueada | "
            "padrão detectado: %r | "
            "primeiros 100 chars: %r",
            match.group()[:50],
            clean[:100],
        )
        raise ValueError(
            "Entrada inválida detectada. "
            "Por favor, reformule sua pergunta sem comandos ao sistema."
        )

    return clean


def validate_file(file_path: str | Path) -> Path:
    """
    Valida um arquivo antes da ingestão.

    Verificações:
    1. Path traversal — arquivo deve estar dentro de docs/
    2. Extensão — apenas extensões permitidas
    3. Existência — arquivo deve existir e ser legível
    4. Não é diretório — deve ser um arquivo regular
    5. Tamanho — dentro do limite configurado
    6. Não vazio — arquivo com conteúdo

    Args:
        file_path: Caminho para o arquivo a validar.

    Returns:
        Path absoluto e resolvido do arquivo.

    Raises:
        ValueError: Para problemas de validação.
        FileNotFoundError: Se o arquivo não existir.
    """
    # resolve() expande symlinks e normaliza o caminho para absoluto.
    # Ex: "docs/../etc/passwd" vira "/etc/passwd" — detectável.
    path = Path(file_path).resolve()

    # Diretório base autorizado. Qualquer arquivo fora dele é bloqueado.
    # resolve() garante que não há truques com "../" ou symlinks.
    docs_dir = Path("docs").resolve()

    try:
        path.relative_to(docs_dir)
    except ValueError:
        raise ValueError(
            f"Acesso negado: '{file_path}' está fora do diretório autorizado.\n"
            f"Coloque o arquivo dentro da pasta 'docs/'."
        )

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: '{path}'")

    if not path.is_file():
        raise ValueError(f"'{path}' não é um arquivo regular.")

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Extensão '{suffix}' não permitida. "
            f"Extensões aceitas: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    size = path.stat().st_size

    if size == 0:
        raise ValueError(f"O arquivo '{path.name}' está vazio.")

    if size > MAX_FILE_SIZE_BYTES:
        size_mb = size / 1024 / 1024
        limit_mb = MAX_FILE_SIZE_BYTES / 1024 / 1024
        raise ValueError(
            f"Arquivo muito grande: {size_mb:.1f} MB. "
            f"Limite: {limit_mb:.0f} MB. "
            f"Ajuste MAX_FILE_SIZE_BYTES no .env para aumentar o limite."
        )

    return path


def compute_file_hash(path: Path) -> str:
    """
    Calcula o hash SHA-256 do arquivo.

    Usado para:
    1. Detectar duplicatas — evita re-ingestão do mesmo arquivo
    2. Detectar adulteração — um arquivo modificado gera hash diferente

    Lê em chunks de 8 KB para não carregar o arquivo inteiro na memória.
    Importante para arquivos grandes (PDFs de centenas de páginas).

    Args:
        path: Caminho do arquivo (já validado por validate_file).

    Returns:
        Hash SHA-256 como string hexadecimal de 64 caracteres.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        # Lê 8 KB por vez — eficiente para qualquer tamanho de arquivo
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def sanitize_chunk_text(text: str) -> str:
    """
    Limpa texto extraído de documentos antes de gerar embeddings.

    Remove:
    - Caracteres de controle (ASCII 0x00–0x1F, exceto \\n e \\t)
    - Caractere DEL (0x7F)
    - Múltiplas quebras de linha consecutivas (mais de 2 → 2)

    Esses caracteres podem:
    - Causar comportamento inesperado no tokenizador do modelo de embedding
    - Ser usados para ocultar texto malicioso visualmente
    - Gerar embeddings inconsistentes

    Args:
        text: Texto bruto extraído do documento.

    Returns:
        Texto limpo, pronto para embedding.
    """
    # Remove caracteres de controle, preservando \\n (0x0A) e \\t (0x09)
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normaliza múltiplas quebras de linha
    clean = re.sub(r"\n{3,}", "\n\n", clean)

    # Remove espaços no início e fim
    return clean.strip()


def validate_response_safety(response: str) -> str:
    """
    Verifica se a resposta do LLM contém informações que não deveria revelar.

    Esta é uma camada adicional de proteção. O Claude normalmente não revela
    o system prompt, mas esta verificação garante que, caso isso aconteça,
    a resposta seja substituída por uma mensagem segura.

    Args:
        response: Texto gerado pelo LLM.

    Returns:
        Resposta original se segura, ou mensagem de fallback.
    """
    # Frases que indicam vazamento do system prompt
    _LEAK_PATTERNS = [
        r"regras\s+obrigatórias",
        r"system\s+prompt",
        r"minhas\s+instruções\s+(são|incluem)",
        r"fui\s+instruído\s+(a|para)",
        r"NUNCA\s+revele",
        r"meu\s+comportamento\s+deve",
    ]

    response_lower = response.lower()
    for pattern in _LEAK_PATTERNS:
        if re.search(pattern, response_lower, re.IGNORECASE):
            logger.warning(
                "Possível vazamento de system prompt detectado na resposta. "
                "Padrão: %r | Resposta (100 chars): %r",
                pattern,
                response[:100],
            )
            return (
                "Não foi possível gerar uma resposta adequada para esta pergunta. "
                "Por favor, tente reformular."
            )

    return response