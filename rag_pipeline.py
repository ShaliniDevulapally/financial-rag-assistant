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
        {"id": r["id"], "content": r["content"], "score": r.get("@search.score")}
        for r in results
    ]

def answer_question(question: str, top_k: int = 5):
    client, _ = get_clients()
    sources = retrieve(question, top_k)
    if not sources:
        return "I couldn't find relevant information in the indexed annual report.", []

    context = "\n\n".join(
        f"[SOURCE {i} | Chunk ID: {s['id']}]\n{s['content']}"
        for i, s in enumerate(sources, 1)
    )
    system = """You are a financial document assistant for JPMorgan Chase's 2025 Annual Report.
Answer ONLY from the supplied retrieved context. Do not invent facts or use outside knowledge.
Give a concise professional answer. If context is insufficient, say so.
Cite supporting passages as [Source 1], [Source 2], etc.
Do not provide investment recommendations."""
    user = f"Question:\n{question}\n\nRetrieved context:\n{context}"
    response = client.chat.completions.create(
        model=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-mini"),
        temperature=0.1,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content, sources
