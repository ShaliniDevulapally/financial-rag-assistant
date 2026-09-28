from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://shalini6720-1993-resource.openai.azure.com/"
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default"
)

client = OpenAI(
    base_url=endpoint.rstrip("/") + "/openai/v1",
    api_key=token_provider
)

# Read the first chunk
with open("annual_report.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunk = text[:1500]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunk
)

embedding = response.data[0].embedding

print("Embedding created successfully!")
print(f"Vector dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")