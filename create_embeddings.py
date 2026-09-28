import json
from pathlib import Path

from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://shalini6720-1993-resource.openai.azure.com/"

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = OpenAI(
    base_url=endpoint.rstrip("/") + "/openai/v1",
    api_key=token_provider
)

# Read the annual report
text = Path("annual_report.txt").read_text(encoding="utf-8")

# Create chunks
chunk_size = 1500
overlap = 200

chunks = []
start = 0

while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]

    if chunk.strip():
        chunks.append(chunk)

    start += chunk_size - overlap

print(f"Total chunks: {len(chunks)}")

documents = []

# Process chunks in batches
batch_size = 50

for start_index in range(0, len(chunks), batch_size):

    batch = chunks[start_index:start_index + batch_size]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch
    )

    for chunk, embedding_data in zip(batch, response.data):
        documents.append({
            "id": len(documents),
            "text": chunk,
            "embedding": embedding_data.embedding
        })

    print(f"Processed {len(documents)}/{len(chunks)} chunks")

# Save embeddings
Path("embeddings.json").write_text(
    json.dumps(documents),
    encoding="utf-8"
)

print("Embedding creation complete!")