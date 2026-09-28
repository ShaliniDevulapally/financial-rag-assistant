from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Your Azure AI Foundry project endpoint
endpoint = "https://shalini6720-1993-resource.services.ai.azure.com/api/projects/shalini6720-1993"

# Use your Azure login instead of an API key"
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default"
)

client = OpenAI(
    base_url=endpoint.rstrip("/") + "/openai/v1",
    api_key=token_provider
)

question  = input("Ask a question:")

response = client.responses.create(
    model="gpt-4.1-mini",
    input=question
)

print(response.output_text)
