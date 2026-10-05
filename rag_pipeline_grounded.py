import os
from typing import Any

from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def get_clients():
    endpoint = _env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    client = OpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=_env("AZURE_OPENAI_API_KEY"),
    )

    search = SearchClient(
        endpoint=_env("AZURE_SEARCH_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX", "financial-documents"),
        credential=AzureKeyCredential(_env("AZURE_SEARCH_KEY")),
    )

    return client, search


def retrieve(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    client, search = get_clients()

    embedding = client.embeddings.create(
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        input=question,
    ).data[0].embedding

    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    results = search.search(
        search_text=question,
        vector_queries=[vector_query],
        select=["id", "content"],
        top=top_k,
    )

    return [
        {
            "id": r["id"],
            "content": r["content"],
            "score": r.get("@search.score"),
        }
        for r in results
    ]


def answer_question(question: str, top_k: int = 5):
    client, _ = get_clients()
    sources = retrieve(question, top_k)

    if not sources:
        return (
            "I couldn't find relevant information in the indexed annual report.",
            [],
        )

    context = "\n\n".join(
        f"[SOURCE {i} | Chunk ID: {s['id']}]\n{s['content']}"
        for i, s in enumerate(sources, 1)
    )

    system = """You are a financial document assistant for JPMorgan Chase's 2025 Annual Report.

Your job is to answer the user's question using ONLY the retrieved source passages supplied below.

GROUNDING RULES — FOLLOW THESE STRICTLY:
1. Do not use outside knowledge, memory, or assumptions.
2. Do not invent, estimate, round, or infer financial figures that are not explicitly supported by the retrieved passages.
3. Every numerical claim must be directly supported by one or more retrieved sources.
4. Never combine, add, subtract, or otherwise calculate figures unless the retrieved text explicitly provides the figures and the calculation is necessary and unambiguous.
5. Carefully distinguish company-wide/firmwide figures from business-segment, product, geographic, or other subgroup figures.
6. Do not combine figures from different tables, sections, periods, or business segments unless the source text explicitly establishes that they belong together.
7. Pay close attention to the reporting period. Do not substitute a 2024 figure for a 2025 figure or vice versa.
8. If sources contain conflicting figures, report the conflict rather than choosing a number yourself.
9. If the retrieved passages do not clearly support the requested answer, say:
   "The retrieved passages do not provide enough information to answer reliably."
10. Keep the answer concise and professional.
11. Cite supporting retrieved passages using [Source 1], [Source 2], etc.
12. Do not provide investment recommendations.

Before finalizing the answer, internally verify that each financial number in your response is traceable to the supplied context and that its scope and reporting period match the question."""

    user = f"""Question:
{question}

Retrieved context:
{context}

Answer the question using the grounding rules above."""

    response = client.chat.completions.create(
        model=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-mini"),
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return response.choices[0].message.content, sources
