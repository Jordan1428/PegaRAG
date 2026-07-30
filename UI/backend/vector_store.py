import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class ChromaVectorStore:
    def __init__(self, persist_dir: str = None):
        if persist_dir is None:
            persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception:
            self.emb_fn = None

    def add_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]) -> int:
        collection_name = f"doc_{doc_id}"
        # Delete old collection if exists to allow clean re-upload
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
            
        col = self.client.create_collection(name=collection_name, embedding_function=self.emb_fn)
        ids = [c["chunk_id"] for c in chunks]
        documents = [c.get("text", c.get("content", "")) for c in chunks]
        metadatas = [{"page_num": c.get("page_num", c.get("page", 1)), "has_table": c.get("has_table", False)} for c in chunks]
        col.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    def search(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            col = self.client.get_collection(name=f"doc_{doc_id}", embedding_function=self.emb_fn)
            count = col.count()
            if count == 0:
                return []
            res = col.query(query_texts=[query], n_results=min(top_k, count))
            if not res or not res.get("documents") or not res["documents"][0]:
                return []
            results = []
            for i, doc_text in enumerate(res["documents"][0]):
                cid = res["ids"][0][i]
                meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                results.append({
                    "chunk_id": cid,
                    "content": doc_text,
                    "page_num": meta.get("page_num", 1),
                    "has_table": meta.get("has_table", False)
                })
            return results
        except Exception as e:
            print(f"[VectorDB Search Error]: {e}")
            return []

    def get_all(self, doc_id: str) -> List[Dict[str, Any]]:
        try:
            col = self.client.get_collection(name=f"doc_{doc_id}", embedding_function=self.emb_fn)
            res = col.get()
            if not res or not res.get("documents"):
                return []
            results = []
            for i, doc_text in enumerate(res["documents"]):
                cid = res["ids"][i]
                meta = res["metadatas"][i] if (res.get("metadatas") and i < len(res["metadatas"])) else {}
                results.append({
                    "chunk_id": cid,
                    "content": doc_text,
                    "page_num": meta.get("page_num", 1) if meta else 1,
                    "has_table": meta.get("has_table", False) if meta else False
                })
            return results
        except Exception as e:
            print(f"[VectorDB get_all Error]: {e}")
            return []
