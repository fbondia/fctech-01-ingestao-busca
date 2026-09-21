"""Busca semantica e geracao de respostas baseadas no PDF."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import Settings, create_vector_store


OUT_OF_CONTEXT_ANSWER = "Não tenho informações necessárias para responder sua pergunta."
RESULTS_LIMIT = 10

PROMPT_TEMPLATE = """CONTEXTO:
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


def format_context(results: list[tuple[object, float]]) -> str:
    """Concatena os documentos recuperados, preservando pagina e score."""
    if not results:
        return "Nenhum conteúdo relevante foi encontrado."

    parts: list[str] = []
    for position, (document, score) in enumerate(results, start=1):
        page = document.metadata.get("page")
        page_label = f", página {page + 1}" if isinstance(page, int) else ""
        parts.append(
            f"[Trecho {position}{page_label}; distância {score:.4f}]\n"
            f"{document.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def search_documents(question: str, *, k: int = RESULTS_LIMIT):
    """Vetoriza a pergunta e devolve os documentos mais proximos."""
    settings = Settings.from_env()
    vector_store = create_vector_store(settings)
    return vector_store.similarity_search_with_score(question, k=k)


def search_prompt(question: str) -> str:
    """Executa a busca RAG e devolve somente o texto final da LLM."""
    question = question.strip()
    if not question:
        raise ValueError("A pergunta nao pode estar vazia.")

    settings = Settings.from_env()
    results = create_vector_store(settings).similarity_search_with_score(
        question,
        k=RESULTS_LIMIT,
    )
    if not results:
        return OUT_OF_CONTEXT_ANSWER

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "contexto": format_context(results),
            "pergunta": question,
        }
    ).strip()
