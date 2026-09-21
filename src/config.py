"""Configuracao compartilhada pelos comandos de ingestao e busca."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "sim", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    embedding_model: str
    chat_model: str
    postgres_connection: str
    collection_name: str
    pdf_path: Path
    recreate_collection: bool

    @classmethod
    def from_env(cls, *, require_api_key: bool = True) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if require_api_key and not api_key:
            raise ValueError(
                "OPENAI_API_KEY nao foi definida. Copie .env.example para .env "
                "e informe uma chave valida."
            )

        raw_pdf_path = Path(os.getenv("PDF_PATH", "document.pdf")).expanduser()
        pdf_path = (
            raw_pdf_path
            if raw_pdf_path.is_absolute()
            else (PROJECT_ROOT / raw_pdf_path).resolve()
        )

        return cls(
            openai_api_key=api_key,
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            postgres_connection=(
                os.getenv("POSTGRES_CONNECTION")
                or os.getenv("DATABASE_URL")
                or "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
            ),
            collection_name=(
                os.getenv("PGVECTOR_COLLECTION")
                or os.getenv("PG_VECTOR_COLLECTION_NAME")
                or "pdf_documents"
            ),
            pdf_path=pdf_path,
            recreate_collection=_as_bool(
                os.getenv("RECREATE_COLLECTION", "true")
            ),
        )


def create_embeddings(settings: Settings):
    """Cria o cliente de embeddings mantendo o modelo em um unico lugar."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def create_vector_store(settings: Settings, *, recreate: bool = False):
    """Conecta ao PGVector e, opcionalmente, recria a collection."""
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=create_embeddings(settings),
        collection_name=settings.collection_name,
        connection=settings.postgres_connection,
        use_jsonb=True,
        pre_delete_collection=recreate,
    )
