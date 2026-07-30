import chromadb
from pathlib import Path

client = chromadb.PersistentClient("data/chromadb")
col = client.get_collection("rag_documents")

print(f"Total Chunks in ChromaDB: {col.count()}")

data = col.get(include=["documents", "metadatas"])
docs = data.get("documents", [])
metas = data.get("metadatas", [])

output_path = Path("data/all_chunks_raw_text.txt")

with open(output_path, "w", encoding="utf-8") as f:
    for idx, (doc, meta) in enumerate(zip(docs, metas)):
        f.write(f"============================================================\n")
        f.write(f"Chunk #{idx + 1} | Metadata: {meta}\n")
        f.write(f"============================================================\n")
        f.write(doc + "\n\n")

print(f"Successfully exported {len(docs)} chunks to {output_path.resolve()}")
