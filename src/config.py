"""
Configurações centralizadas do projeto.

ATENÇÂO: 
Todos os valores vem daqui, por tanto nenhum outro módulo deve chamar os.getenv() ou load_dotenv().

"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Carrega o .env da raiz do projeto, independentemente de onde o script é executado
load_dotenv(Path(__file__).parent.parent / ".env")


def _require(key: str) -> str:
    """
    Lê uma variável de ambiente obrigatória.
    Falha imediatamente (fail-fast) com mensagem clara se ausente.

    Por que a utilização de fail-fast?
    Sem isso, o sistema funcionaria até o momento de usar a chave
    (ex: ao chamar a API pela primeira vez), gerando um erro confuso
    dentro de uma função de negócio. Assim, o erro aparece na inicialização.
    """
    value = os.getenv(key, "").strip()
    if not value:
        print(
            f"\nERRO: Variável de ambiente '{key}' não encontrada.\n"
            f"   1. Copie .env.example para .env\n"
            f"   2. Preencha '{key}' com o valor correto\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


# APIs obrigatórias ────────────────────────────────────────────────────────
OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str = _require("ANTHROPIC_API_KEY")

# Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")

# Modelos ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS: int = 1536   # fixo para text-embedding-3-small
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-haiku-4-5")

# Ingestão ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150
ALLOWED_EXTENSIONS: frozenset = frozenset({".pdf", ".txt", ".md"})

# Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MIN_SCORE_THRESHOLD: float = float(os.getenv("MIN_SCORE_THRESHOLD", "0.75"))

# Segurança ────────────────────────────────────────────────────────────────
MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000")) 