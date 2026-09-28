from pathlib import Path

text = Path("annual_report.txt").read_text(encoding="utf-8")

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

print(f"Total characters: {len(text):,}")
print(f"Total chunks: {len(chunks):,}")

Path("chunks.txt").write_text(
    "\n\n--- CHUNK ---\n\n".join(chunks),
    encoding="utf-8"
)

print("Chunking complete!")