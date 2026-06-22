"""
Testes para src/chain.py.

Estratégia: mockar retriever e cliente Anthropic para testar
apenas a lógica de chain.py sem depender de APIs externas.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestChain:

    @patch("chain._client")
    @patch("chain.retrieve")
    def test_resposta_normal_com_contexto(self, mock_retrieve, mock_client):
        """Com chunks relevantes, deve retornar resposta com fontes."""
        # Arrange
        mock_retrieve.return_value = [
            {"text": "O prazo é 30 dias.", "source": "contrato.pdf", "score": 0.92},
            {"text": "Exceções estão na cláusula 4.2.", "source": "contrato.pdf", "score": 0.88},
        ]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="O prazo de entrega é 30 dias.")]
        mock_client.messages.create.return_value = mock_response

        from chain import ask

        # Act
        result = ask("Qual é o prazo de entrega?")

        # Assert
        assert result["answer"] == "O prazo de entrega é 30 dias."
        assert "contrato.pdf" in result["sources"]
        assert result["chunks_used"] == 2
        assert result["no_context"] is False

    @patch("chain._client")
    @patch("chain.retrieve")
    def test_sem_contexto_retorna_no_context_true(self, mock_retrieve, mock_client):
        """Sem chunks relevantes, no_context deve ser True."""
        mock_retrieve.return_value = []  # nenhum chunk encontrado
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Não encontrei essa informação.")]
        mock_client.messages.create.return_value = mock_response

        from chain import ask

        result = ask("Pergunta sobre algo não indexado")

        assert result["no_context"] is True
        assert result["chunks_used"] == 0
        assert result["sources"] == []

    @patch("chain._client")
    @patch("chain.retrieve")
    def test_fontes_duplicadas_sao_deduplicadas(self, mock_retrieve, mock_client):
        """Mesmo arquivo aparecendo em múltiplos chunks deve aparecer uma vez."""
        mock_retrieve.return_value = [
            {"text": "Trecho 1", "source": "manual.pdf", "score": 0.95},
            {"text": "Trecho 2", "source": "manual.pdf", "score": 0.90},
            {"text": "Trecho 3", "source": "manual.pdf", "score": 0.85},
        ]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Resposta baseada no manual.")]
        mock_client.messages.create.return_value = mock_response

        from chain import ask

        result = ask("Pergunta sobre o manual")

        assert result["sources"].count("manual.pdf") == 1
        assert len(result["sources"]) == 1

    @patch("chain._client")
    @patch("chain.retrieve")
    @patch("chain.validate_response_safety")
    def test_resposta_suspeita_e_validada(self, mock_validate, mock_retrieve, mock_client):
        """validate_response_safety deve ser chamado com a resposta bruta."""
        mock_retrieve.return_value = []
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="resposta bruta")]
        mock_client.messages.create.return_value = mock_response
        mock_validate.return_value = "resposta filtrada"

        from chain import ask

        result = ask("pergunta qualquer")

        mock_validate.assert_called_once_with("resposta bruta")
        assert result["answer"] == "resposta filtrada"