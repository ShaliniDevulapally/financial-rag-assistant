# Financial Document AI Assistant

A Retrieval-Augmented Generation (RAG) application for grounded question answering over JPMorgan Chase's 2025 Annual Report.

## Architecture
JPMorgan Annual Report → chunking → text-embedding-3-small → Azure AI Search (vector + keyword retrieval) → gpt-4.1-mini → grounded answer + source references → Streamlit.

## Stack
Python, Azure OpenAI/Microsoft Foundry, text-embedding-3-small, gpt-4.1-mini, Azure AI Search, vector search, hybrid retrieval, Streamlit.

## Indexed corpus
364-page JPMorgan Chase 2025 Annual Report; 996 text chunks; 1,536-dimensional embeddings.

## Configuration
Set: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_CHAT_DEPLOYMENT, AZURE_EMBEDDING_DEPLOYMENT, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX.

Never commit API keys, Search admin keys, .env files containing secrets, PDFs, or large generated embedding files.

## Run
pip install -r requirements.txt
streamlit run app.py

## Example questions
- What were JPMorgan Chase's major sources of revenue in 2025?
- What were the company's major business segments?
- What factors affected net income?
- What does the annual report say about credit quality?
- How did the company describe its capital position?

## Limitations
Answers are grounded in the indexed annual-report corpus and are informational only, not investment advice.
