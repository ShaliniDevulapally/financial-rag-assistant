import json
import numpy as np

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

# Load our document embeddings
with open("embeddings.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

question = input("Ask about JPMorgan's annual report: ")

# Create an embedding for the question
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

question_vector = np.array(response.data[0].embedding)

# Compare question with every document chunk
results = []

for document in documents:
    document_vector = np.array(document["embedding"])

    similarity = np.dot(question_vector, document_vector) / (
        np.linalg.norm(question_vector) *
        np.linalg.norm(document_vector)
    )

    results.append((similarity, document["text"]))

# Highest similarity first
results.sort(reverse=True)

# Get the top 5 relevant chunks
top_results = results[:5]

# Build context from the retrieved chunks
context = "\n\n".join(
    f"Source {i + 1}:\n{text}"
    for i, (score, text) in enumerate(top_results)
)

# Ask GPT to answer using only the retrieved information
prompt = f"""
You are a financial document assistant.

Answer the user's question using ONLY the information
provided in the annual report excerpts below.

If the excerpts do not contain enough information,
say that the excerpts do not provide enough information.

Annual report excerpts:

{context}

User question:

{question}
"""

answer = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt
)

print("\nAnswer:\n")
print(answer.output_text)