import chromadb

client = chromadb.PersistentClient("data/chromadb")
col = client.get_collection("rag_documents")
data = col.get(include=["documents", "metadatas"])

print(f"Total Chunks in ChromaDB: {len(data['ids'])}")

for idx, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
    lines = [line.strip() for line in doc.split("\n") if line.strip()]
    first_line = lines[0] if lines else "Empty"
    print(f"Chunk #{idx+1} | Length: {len(doc)} chars | Lines: {len(lines)} | First line: {first_line[:80]}")
