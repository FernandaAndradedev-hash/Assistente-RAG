"""
Retrieval: busca os chunks mais relevantes para uma pergunta.

Responsabilidade única: receber uma pergunta já sanitizada e retornar
os chunks do banco vetorial que melhor respondem a ela.

Este módulo não conhece o LLM nem o formato final da resposta —
essa é responsabilidade do chain.py.
"""
import logging

from openai import OpenAI
from qdrant_client import QdrantClient

import config

logger = logging.getLogger(__name__)

_openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
_qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    
    k = top_k or config.RETRIEVAL_TOP_K

    # Gera embedding da pergunta com o MESMO modelo usado na ingestão
    response = _openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=query,
    )
    query_vector = response.data[0].embedding

    # Busca no Qdrant
    # score_threshold descarta resultados com baixa similaridade —
    # melhor retornar "não encontrado" do que contexto irrelevante.
    search_results = _qdrant_client.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=k,
        with_payload=True,
        score_threshold=config.MIN_SCORE_THRESHOLD,
    )

    chunks = [
        {
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "score": round(hit.score, 4),
        }
        for hit in search_results
    ]

    if not chunks:
        logger.info(
            "Nenhum chunk encontrado acima do threshold %.2f para: '%s...'",
            config.MIN_SCORE_THRESHOLD,
            query[:50],
        )
    else:
        logger.debug(
            "%d chunks recuperados para '%s...' | scores: %s",
            len(chunks),
            query[:40],
            [c["score"] for c in chunks],
        )

    return chunks