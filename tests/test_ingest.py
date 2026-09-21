from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import ingest  # noqa: E402


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)
        return [str(index) for index in range(len(documents))]


class IngestTests(unittest.TestCase):
    def test_pdf_is_split_into_expected_chunk_size(self):
        vector_store = FakeVectorStore()
        env = {
            "OPENAI_API_KEY": "test-key",
            "PDF_PATH": "document.pdf",
            "RECREATE_COLLECTION": "true",
        }

        with patch.dict(os.environ, env, clear=False), patch.object(
            ingest,
            "create_vector_store",
            return_value=vector_store,
        ) as create_store:
            total = ingest.ingest_pdf()

        self.assertEqual(total, len(vector_store.documents))
        self.assertGreater(total, 0)
        self.assertTrue(
            all(
                0 < len(document.page_content) <= ingest.CHUNK_SIZE
                for document in vector_store.documents
            )
        )
        self.assertEqual(
            [document.metadata["chunk"] for document in vector_store.documents],
            list(range(total)),
        )
        create_store.assert_called_once()
        self.assertTrue(create_store.call_args.kwargs["recreate"])


if __name__ == "__main__":
    unittest.main()
