"""Carrega o PDF, cria chunks e grava seus embeddings no PostgreSQL."""

from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, create_vector_store


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def ingest_pdf() -> int:
    settings = Settings.from_env()
    if not settings.pdf_path.is_file():
        raise FileNotFoundError(f"PDF nao encontrado: {settings.pdf_path}")

    print(f"Lendo PDF: {settings.pdf_path}")
    pages = PyPDFLoader(str(settings.pdf_path)).load()
    if not pages:
        raise ValueError("O PDF nao possui paginas legiveis.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = [
        chunk
        for chunk in splitter.split_documents(pages)
        if chunk.page_content.strip()
    ]
    if not chunks:
        raise ValueError("Nenhum texto foi extraido do PDF.")

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index
        chunk.metadata["source"] = str(settings.pdf_path)

    vector_store = create_vector_store(
        settings,
        recreate=settings.recreate_collection,
    )
    vector_store.add_documents(chunks)

    print(
        f"Ingestao concluida: {len(pages)} pagina(s), {len(chunks)} chunk(s), "
        f"collection '{settings.collection_name}'."
    )
    return len(chunks)


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as exc:
        raise SystemExit(f"Erro na ingestao: {exc}") from exc
