from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import search  # noqa: E402


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, question, *, k):
        self.calls.append((question, k))
        return self.results


class SearchTests(unittest.TestCase):
    def test_formats_context_with_page_and_score(self):
        context = search.format_context(
            [
                (
                    Document(
                        page_content="Faturamento: R$ 10 milhões",
                        metadata={"page": 2},
                    ),
                    0.12,
                )
            ]
        )

        self.assertIn("página 3", context)
        self.assertIn("0.1200", context)
        self.assertIn("Faturamento: R$ 10 milhões", context)

    def test_returns_fixed_answer_when_search_has_no_results(self):
        store = FakeVectorStore([])
        with patch.object(
            search.Settings,
            "from_env",
            return_value=object(),
        ), patch.object(search, "create_vector_store", return_value=store):
            answer = search.search_prompt("Uma pergunta sem resultados")

        self.assertEqual(answer, search.OUT_OF_CONTEXT_ANSWER)
        self.assertEqual(store.calls, [("Uma pergunta sem resultados", 10)])

    def test_rejects_empty_question_before_accessing_services(self):
        with self.assertRaisesRegex(ValueError, "vazia"):
            search.search_prompt("   ")


if __name__ == "__main__":
    unittest.main()
