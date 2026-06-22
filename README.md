# Assistente RAG — Chatbot de Documentos Internos
 
> Chatbot inteligente que responde perguntas sobre documentos da empresa usando **Retrieval-Augmented Generation (RAG)** — implementando diretrizes de Segurança e Alinhamento de LLMs (LLM Safety & Alignment)
 
---
 
## Sobre o projeto
 
Este projeto implementa um assistente de IA capaz de responder perguntas com base em documentos internos (PDFs, TXTs, MDs) carregados pelo usuário. Em vez de depender do conhecimento geral do modelo de linguagem, o sistema busca as informações diretamente nos documentos indexados — garantindo respostas precisas e rastreáveis.
 
A arquitetura combina **busca semântica** com **geração de linguagem natural**, formando um pipeline RAG completo com múltiplas camadas de segurança.
 
---
 
## Funcionalidades
 
- Ingestão de arquivos PDF, TXT e MD com controle de duplicidade via assinatura hash SHA-256
- Busca semântica por similaridade de cosseno no Qdrant
- Geração de respostas baseadas em contexto utilizando a API do Claude (Anthropic) com citação estruturada de fontes
- Camada de segurança ativa contra injeção de prompt (Prompt Injection), Path Traversal e envenenamento de dados (Data Poisoning)
- API REST documentada com FastAPI e Swagger UI
- Interface de chat com Chainlit e design system customizado
- Testes unitários com cobertura de segurança
---
 
## Stack
 
| Camada | Tecnologia | Função |
|--------|-----------|--------|
| Embedding | OpenAI `text-embedding-3-small` | Transforma texto em vetores numéricos |
| Vector Store | Qdrant (Docker) | Armazena e busca vetores por similaridade |
| LLM | Anthropic Claude Haiku | Gera respostas baseadas no contexto |
| API | FastAPI | Endpoints REST com validação automática |
| Interface | Chainlit | UI de chat com CSS customizado |
| Segurança | `validators.py` | Sanitização, detecção de injection e guardrails |
 
---
 
## Arquitetura
 
```
┌─────────────────── INGESTÃO ────────────────────────┐
│  PDF/TXT → Validação → Chunking → Embedding → Qdrant │
└──────────────────────────────────────────────────────┘
 
┌─────────────────── CONSULTA ────────────────────────┐
│  Pergunta → Sanitização → Embedding → Busca Qdrant  │
│  → Contexto → Prompt seguro → Claude → Resposta     │
└──────────────────────────────────────────────────────┘
```
 
---
 
## Segurança
 
Este projeto implementa quatro camadas de proteção:
 
| Ameaça | Proteção |
|--------|----------|
| Prompt Injection | Detecção via regex em `validators.py` + guardrails no system prompt |
| Envenenamento de dados | Hash SHA-256 + sanitização de chunks + instrução no system prompt |
| Path Traversal | `Path.resolve()` com verificação de diretório autorizado |
| Vazamento de system prompt | Validação da resposta do LLM antes de retornar ao usuário |
 
---
 
## Como rodar localmente
 
### Pré-requisitos
 
- Python 3.11+
- Docker Desktop
- Chaves de API: OpenAI e Anthropic
### Instalação
 
```bash
# 1. Clone o repositório
git clone https://github.com/FernandaAndradedev-hash/Assistente-RAG.git
cd Assistente-RAG
 
# 2. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # Mac/Linux
 
# 3. Instale as dependências
pip install -r requirements.txt
 
# 4. Configure as variáveis de ambiente
copy .env.example .env
# Abra o .env e preencha OPENAI_API_KEY e ANTHROPIC_API_KEY
 
# 5. Suba o Qdrant
docker-compose up -d
```
 
### Ingestão de documentos
 
```bash
# Coloque seus PDFs na pasta docs/ e rode:
cd src
python ingest.py
```
 
### Rodando a aplicação
 
```bash
# Terminal 1 — API
cd src
uvicorn api:app --reload --port 8000
 
# Terminal 2 — Interface
cd interface
chainlit run app.py --port 8001
```
 
| Serviço | URL |
|---------|-----|
| Interface de chat | http://localhost:8001 |
| Documentação da API (Swagger) | http://localhost:8000/docs |
| Dashboard do Qdrant | http://localhost:6333/dashboard |
 
---
 
## Testes
 
```bash
pytest tests/ -v
```
 
Os testes cobrem:
- Sanitização de entrada (HTML, espaços, unicode)
- Detecção de 30+ variantes de prompt injection
- Bloqueio de path traversal
- Validação de arquivos (extensão, tamanho, conteúdo vazio)
- Lógica do chain com mocks das APIs externas
---
 
## Estrutura do projeto
 
```
Assistente-RAG/
├── src/
│   ├── config.py        # Configurações centralizadas
│   ├── validators.py    # Camada de segurança
│   ├── ingest.py        # Pipeline de ingestão
│   ├── retriever.py     # Busca semântica
│   ├── chain.py         # Orquestração do LLM
│   └── api.py           # Endpoints FastAPI
├── interface/
│   ├── app.py           # Lógica do chat
│   └── public/
│       └── custom.css   # Design system
├── tests/               # Testes unitários
├── docs/                # Coloque seus PDFs aqui
├── docker-compose.yml
└── requirements.txt
```
 
---
 
## Documentação completa
 
O guia de construção detalhado está em [`docs-guide/`](docs-guide/), cobrindo:
 
- Conceitos de RAG, embeddings e vector stores
- Decisões de arquitetura e escolha de tecnologias
- Implementação de cada camada de segurança
- Design system completo da interface
- Estratégia de testes
---
 
## Licença
 
Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.