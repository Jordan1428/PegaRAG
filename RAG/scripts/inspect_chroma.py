import sys
import argparse
from pathlib import Path
import chromadb

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def inspect_chroma(show_all=False, limit=5, export=False):
    db_path = str(config.CHROMADB_DIR)
    print("=" * 60)
    print(f"ChromaDB Chunk Text Inspector")
    print(f"Database Path: {db_path}")
    print("=" * 60)

    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return

    if not collections:
        print("\n⚠️ No collections found. Please run 'python scripts/01_build_index.py' first.")
        return

    target_name = config.COLLECTION_NAME
    try:
        collection = client.get_collection(target_name)
    except Exception:
        collection = collections[0]

    count = collection.count()
    print(f"\nCollection Name: '{collection.name}' | Total Chunks: {count}")

    if count == 0:
        print("Collection is empty.")
        return

    # Retrieve all documents and metadata
    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    display_limit = count if show_all else min(limit, count)

    print(f"\n--- Displaying Original Text for {display_limit} Chunk(s) ---\n")

    output_lines = []

    for idx in range(display_limit):
        chunk_id = ids[idx]
        meta = metas[idx] if idx < len(metas) else {}
        doc_text = docs[idx] if idx < len(docs) else ""

        header = f"🔹 [Chunk #{idx + 1}/{count}] ID: {chunk_id} | Source: {meta.get('source', 'Unknown')}"
        separator = "-" * 60
        
        print(header)
        print(f"Metadata: {meta}")
        print(f"Original Text:")
        print(doc_text)
        print(separator + "\n")

        output_lines.append(header)
        output_lines.append(f"Metadata: {meta}")
        output_lines.append(f"Original Text:\n{doc_text}")
        output_lines.append(separator + "\n")

    if export:
        export_file = config.BASE_DIR / "data" / "all_chunks_raw_text.txt"
        export_file.write_text("\n".join(output_lines), encoding="utf-8")
        print(f"✅ Successfully exported all chunk texts to: {export_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect original chunk texts in ChromaDB.")
    parser.add_argument("--all", "-a", action="store_true", help="Print all chunks' original text.")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Number of chunks to display (default: 5).")
    parser.add_argument("--export", "-e", action="store_true", help="Export output to a text file data/all_chunks_raw_text.txt.")
    args = parser.parse_args()

    inspect_chroma(show_all=args.all, limit=args.limit, export=args.export)
