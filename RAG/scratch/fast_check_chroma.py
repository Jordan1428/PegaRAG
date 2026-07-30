import chromadb

client = chromadb.PersistentClient("data/chromadb")
try:
    col = client.get_collection("rag_documents")
    data = col.get(include=["documents", "metadatas"])
    print(f"Total Chunks in ChromaDB: {len(data['ids'])}")
    for i, doc in enumerate(data['documents']):
        lines = [l.strip() for l in doc.split('\n') if l.strip()]
        title = lines[0] if lines else ""
        print(f"Chunk #{i+1}: {title[:80]}")
except Exception as e:
    print(f"Error: {e}")
