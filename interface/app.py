"""
Lógica do chat com Chainlit.

Este arquivo orquestra a experiência do usuário:
- Mensagem de boas-vindas com sugestões
- Processamento de cada mensagem
- Exibição das fontes
- Tratamento de erros de forma amigável
"""
import sys
from pathlib import Path

import chainlit as cl

# Adiciona src/ ao path para importar os módulos do projeto
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chain import ask
from validators import sanitize_query



# Configurações de apresentação ───────────────────────────────────────────────

ASSISTANT_NAME = "Assistente RAG"
WELCOME_MESSAGE = """ Olá! Sou o **Assistente de Documentos Internos**.

Posso responder perguntas com base nos documentos que foram carregados no sistema.

**Exemplos do que posso responder:**
- "Qual é o prazo de entrega descrito no contrato?"
- "Quais são as políticas de reembolso?"
- "Resuma os pontos principais do relatório de Q3"

*Importante: minha base de conhecimento é limitada aos documentos indexados.*"""

SUGGESTIONS = [
    "Quais documentos estão disponíveis?",
    "Qual é a política de reembolso?",
    "Resuma o contrato principal",
]



# Eventos do Chainlit ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """
    Executado quando o usuário abre uma nova conversa.
    Exibe a mensagem de boas-vindas e configura o estado inicial.
    """
    # Envia mensagem de boas-vindas
    await cl.Message(
        content=WELCOME_MESSAGE,
        author=ASSISTANT_NAME,
    ).send()

    # Armazena estado da conversa na sessão do Chainlit
    # cl.user_session é um dicionário por usuário/aba
    cl.user_session.set("message_count", 0)


@cl.on_message
async def on_message(message: cl.Message):
    """
    Executado a cada mensagem enviada pelo usuário.

    Fluxo:
    1. Sanitiza a entrada
    2. Mostra indicador de "digitando"
    3. Chama o chain (retrieval + LLM)
    4. Exibe resposta com fontes
    """
    count = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", count)

    # Etapa 1: Sanitização
    try:
        clean_query = sanitize_query(message.content)
    except ValueError as exc:
        await cl.Message(
            content=f"⚠️ **Entrada inválida:** {str(exc)}",
            author=ASSISTANT_NAME,
        ).send()
        return

    # Etapa 2: Indicador de processamento
    # cl.Step cria um bloco colapsável que mostra o que está acontecendo
    async with cl.Step(name="🔍 Buscando nos documentos...") as step:
        try:
            result = ask(clean_query)
            step.output = f"Encontrados {result['chunks_used']} trechos relevantes."
        except Exception as exc:
            step.output = f"Erro: {str(exc)}"
            await cl.Message(
                content=(
                    " Ocorreu um erro ao processar sua pergunta. "
                    "Por favor, tente novamente."
                ),
                author=ASSISTANT_NAME,
            ).send()
            return

    # Etapa 3: Formata e exibe a resposta

    # Aviso se não encontrou contexto
    prefix = ""
    if result["no_context"]:
        prefix = (
            "⚠️ *Nenhum documento relevante encontrado. "
            "A resposta abaixo é baseada em conhecimento geral.*\n\n"
        )

    # Formata as fontes como lista no final da mensagem
    sources_text = ""
    if result["sources"]:
        sources_list = "\n".join(f"  - `{src}`" for src in result["sources"])
        sources_text = f"\n\n---\n**📄 Fontes:**\n{sources_list}"

    # Envia a resposta completa
    await cl.Message(
        content=prefix + result["answer"] + sources_text,
        author=ASSISTANT_NAME,
    ).send()