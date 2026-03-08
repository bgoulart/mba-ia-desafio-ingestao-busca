# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema RAG (Retrieval-Augmented Generation) que ingere um PDF, armazena os embeddings no PostgreSQL com pgvector e responde perguntas via chat usando OpenAI.

## Requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API da OpenAI

## Passo a passo

### 1. Clone o repositório e entre na pasta

```bash
git clone <url-do-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Configure as variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env` com seus valores — **todas as variáveis são obrigatórias**:

```env
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5-nano
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documents
PDF_PATH=./document.pdf
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

> O sistema valida todas as variáveis na inicialização e lança `ValueError` imediatamente caso alguma esteja ausente ou vazia.

### 3. Suba o banco de dados

```bash
docker compose up -d
```

Isso inicializa o PostgreSQL com a extensão `pgvector` habilitada. Aguarde alguns segundos até o container estar saudável.

### 4. Crie o ambiente virtual e instale as dependências

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# ou
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 5. Ingira o PDF

Execute o script de ingestão para carregar o PDF, dividir em chunks e armazenar os embeddings no banco:

```bash
python src/ingest.py
```

O processo vai:

1. Carregar o PDF definido em `PDF_PATH`
2. Dividir o conteúdo em chunks conforme `CHUNK_SIZE` e `CHUNK_OVERLAP`
3. Gerar embeddings via OpenAI (`OPENAI_EMBEDDING_MODEL`)
4. Persistir os vetores no PostgreSQL

### 6. Inicie o chat

```bash
python src/chat.py
```

Digite suas perguntas sobre o conteúdo do PDF. O sistema busca os trechos mais relevantes e gera respostas baseadas apenas no conteúdo ingerido.

Para encerrar, digite `sair`, `exit` ou `quit`.

## Estrutura do projeto

```
.
├── docker-compose.yml   # PostgreSQL com pgvector
├── document.pdf         # PDF de exemplo
├── requirements.txt     # Dependências Python
├── .env.example         # Template de variáveis de ambiente
└── src/
    ├── ingest.py        # Ingestão do PDF e geração de embeddings
    ├── search.py        # Busca por similaridade e prompt RAG
    └── chat.py          # Interface de chat interativo
```

## Variáveis de ambiente

Todas as variáveis são obrigatórias. O sistema lança `ValueError` na inicialização caso alguma esteja ausente ou vazia.


| Variável                    | Descrição                                           |
| --------------------------- | --------------------------------------------------- |
| `OPENAI_API_KEY`            | Chave de API da OpenAI                              |
| `OPENAI_EMBEDDING_MODEL`    | Modelo de embeddings (ex: `text-embedding-3-small`) |
| `OPENAI_CHAT_MODEL`         | Modelo de chat (ex: `gpt-5-nano`)                   |
| `DATABASE_URL`              | URL de conexão com o PostgreSQL                     |
| `PG_VECTOR_COLLECTION_NAME` | Nome da coleção no pgvector                         |
| `PDF_PATH`                  | Caminho para o arquivo PDF a ser ingerido           |
| `CHUNK_SIZE`                | Tamanho dos chunks em tokens                        |
| `CHUNK_OVERLAP`             | Sobreposição entre chunks em tokens                 |


