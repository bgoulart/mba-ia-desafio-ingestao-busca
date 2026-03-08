import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} não está definido. Configure no .env.")
    return value


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _normalize_database_url(url: str) -> str:
    if not url:
        return url
    if url.startswith("postgresql://") and "+psycopg" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def search_prompt():
    """Retorna um callable que recebe uma pergunta e retorna a resposta da LLM."""
    database_url = _require_env("DATABASE_URL")
    collection_name = _require_env("PG_VECTOR_COLLECTION_NAME")
    openai_api_key = _require_env("OPENAI_API_KEY")
    chat_model = _require_env("OPENAI_CHAT_MODEL")
    embedding_model = _require_env("OPENAI_EMBEDDING_MODEL")

    connection = _normalize_database_url(database_url)

    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key)
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    llm = ChatOpenAI(model=chat_model, api_key=openai_api_key)

    def chain(pergunta: str) -> str:
        docs = vector_store.similarity_search(pergunta, k=10)
        contexto = "\n\n".join(doc.page_content for doc in docs)
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
        response = llm.invoke(prompt)
        return response.content

    return chain
