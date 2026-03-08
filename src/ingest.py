import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} não está definido. Configure no .env.")
    return value


PDF_PATH = _require_env("PDF_PATH")
DATABASE_URL = _require_env("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = _require_env("PG_VECTOR_COLLECTION_NAME")
CHUNK_SIZE = int(_require_env("CHUNK_SIZE"))
CHUNK_OVERLAP = int(_require_env("CHUNK_OVERLAP"))


def _normalize_database_url(url: str) -> str:
    """Garante que a URL use o driver psycopg para langchain-postgres."""
    if not url:
        return url
    if url.startswith("postgresql://") and "+psycopg" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _get_embeddings():
    """Retorna o cliente de embeddings OpenAI."""
    model = _require_env("OPENAI_EMBEDDING_MODEL")
    api_key = _require_env("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=model, api_key=api_key)


def ingest_pdf():
    """Carrega o PDF, divide em chunks, gera embeddings e persiste no pgvector."""
    path = Path(PDF_PATH)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {path}")

    connection = _normalize_database_url(DATABASE_URL)

    logger.info("Carregando PDF: %s", path)
    loader = PyPDFLoader(str(path))
    documents = loader.load()

    if not documents:
        logger.warning("Nenhum conteúdo extraído do PDF.")
        return

    logger.info("Dividindo em chunks (size=%s, overlap=%s).", CHUNK_SIZE, CHUNK_OVERLAP)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False
    )
    chunks = splitter.split_documents(documents)
    logger.info("Gerados %s chunks.", len(chunks))

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in chunks
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = _get_embeddings()

    logger.info("Conectando ao pgvector (collection=%s).", PG_VECTOR_COLLECTION_NAME)
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=connection,
        use_jsonb=True,
    )
    vector_store.add_documents(enriched, ids=ids)
    logger.info("Ingestão concluída: %s documentos adicionados ao pgvector.", len(enriched))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_pdf()
