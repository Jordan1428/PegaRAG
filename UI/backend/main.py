import os
import sys
import uuid
import logging
import uvicorn
import chromadb
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Ensure RAG package root is in sys.path
RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RAG"))
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from src import config
from src.ingestion.parser import parse_and_chunk_document
from src.ingestion.indexer import build_and_save_index, get_retriever
from src.pipeline.graph import build_rag_graph
from src.llm_factory import get_llm

logger = logging.getLogger("ui_backend")

app = FastAPI(title="AI Document Chat System Backend (Integrated RAG)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatReq(BaseModel):
    document_id: Optional[Any] = ""
    question: Optional[Any] = ""

def extract_tokens(res):
    if not res:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    u = {}
    if hasattr(res, "usage_metadata") and res.usage_metadata:
        u = res.usage_metadata
    elif hasattr(res, "response_metadata") and res.response_metadata:
        u = res.response_metadata.get("token_usage", {}) or res.response_metadata.get("usage", {})
    inp = u.get("input_tokens", u.get("prompt_tokens", 0))
    out = u.get("output_tokens", u.get("completion_tokens", 0))
    tot = u.get("total_tokens", inp + out)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}

def handle_map_reduce_summary(doc_id: str) -> Dict[str, Any]:
    """Map-Reduce strategy for document summary as specified in Spec Section 5.2."""
    db_path = str(config.CHROMADB_DIR)
    chroma_client = chromadb.PersistentClient(path=db_path)
    try:
        col = chroma_client.get_collection(name=doc_id)
        res = col.get()
    except Exception as e:
        return {"answer": f"未找到 Document ID ({doc_id}) 的文件內容。", "source_chunks": [], "chunks_details": [], "token_usage": {}}

    documents = res.get("documents", [])
    ids = res.get("ids", [])
    metadatas = res.get("metadatas", [])

    if not documents:
        return {"answer": "文件中未找到可供摘要的內容。", "source_chunks": [], "chunks_details": [], "token_usage": {}}

    llm = get_llm()

    # Step 1: Map Stage - Select representative chunks across the document
    step = max(1, len(documents) // 10)
    selected_indices = list(range(0, len(documents), step))[:10]

    map_summaries = []
    source_ids = []
    chunks_details = []
    total_inp, total_out = 0, 0

    for idx in selected_indices:
        doc_text = documents[idx]
        cid = ids[idx] if idx < len(ids) else f"chunk_{idx+1:03d}"
        meta = metadatas[idx] if (metadatas and idx < len(metadatas)) else {}
        page_num = meta.get("page_num", meta.get("page", 1)) if meta else 1

        source_ids.append(str(cid))
        chunks_details.append({
            "chunk_id": str(cid),
            "content": doc_text,
            "page_num": page_num
        })

        map_prompt = f"Summarize this section of the document concisely:\n\n{doc_text[:1500]}"
        try:
            response = llm.invoke(map_prompt)
            summary_text = response.content if hasattr(response, "content") else str(response)
            t_data = extract_tokens(response)
            total_inp += t_data["input_tokens"]
            total_out += t_data["output_tokens"]
        except Exception:
            summary_text = doc_text[:200] + "..."

        map_summaries.append(f"• [{cid}] (Page {page_num}): {summary_text}")

    # Step 2: Reduce Stage - Synthesize into final overall summary (Few-Shot Dynamic Prompt)
    combined = "\n\n".join(map_summaries)
    reduce_prompt = (
        "Synthesize the section summaries into a structured summary. "
        "Self-determine 3-4 key focus dimensions suited for the document type, and summarize directly under Markdown headings.\n"
        "Do NOT print any meta-header lines such as 'Document Type:' or 'Focus Dimensions:'. Start directly with the Markdown headings.\n\n"
        "Example 1 (Paper):\n"
        "Input: [Chunk 1]: LightRAG dual-level retrieval. [Chunk 2]: Outperforms GraphRAG with 99% cost reduction.\n"
        "Output:\n"
        "### Core Innovation\n"
        "- Combines graph indexing with dual-level retrieval for fast search.\n\n"
        "### Performance & Cost\n"
        "- Outperforms GraphRAG while reducing API cost by 99%.\n\n"
        "Example 2 (Contract):\n"
        "Input: [Chunk 1]: NDA for source code. [Chunk 2]: 3-year obligation; breach leads to damages.\n"
        "Output:\n"
        "### Scope\n"
        "- Protects proprietary source code and trade secrets.\n\n"
        "### Obligations & Breach\n"
        "- 3-year confidentiality; breach triggers financial damages.\n\n"
        "Task:\n"
        "Synthesize the document below following the exact pattern above:\n\n"
        f"{combined}"
    )

    try:
        resp = llm.invoke(reduce_prompt)
        final_answer = resp.content if hasattr(resp, "content") else str(resp)
        t_data = extract_tokens(resp)
        total_inp += t_data["input_tokens"]
        total_out += t_data["output_tokens"]
    except Exception as e:
        final_answer = f"摘要生成失敗: {str(e)}"

    return {
        "answer": final_answer,
        "source_chunks": source_ids,
        "chunks_details": chunks_details,
        "token_usage": {
            "input_tokens": total_inp,
            "output_tokens": total_out,
            "total_tokens": total_inp + total_out
        }
    }

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    force_reparse: bool = Form(False)
):
    if not file.filename.lower().endswith(".pdf"):
        return {"status": "error", "message": "無法解析此PDF文件，僅支援 .pdf 檔案"}
    
    try:
        content = await file.read()
        import hashlib
        file_hash = hashlib.md5(content).hexdigest()[:8]
        doc_id = f"doc_{file_hash}"
        
        # Check if collection already exists in persistent ChromaDB
        db_path = str(config.CHROMADB_DIR)
        chroma_client = chromadb.PersistentClient(path=db_path)

        if force_reparse:
            try:
                chroma_client.delete_collection(name=doc_id)
                logger.info(f"Force reparse requested: Deleted existing collection '{doc_id}'.")
            except Exception:
                pass
        else:
            try:
                col = chroma_client.get_collection(name=doc_id)
                cnt = col.count()
                if cnt > 0:
                    logger.info(f"Document {doc_id} already exists in ChromaDB ({cnt} chunks). Using persistent cache.")
                    return {"document_id": doc_id, "status": "success", "chunk_count": cnt, "cached": True}
            except Exception:
                pass

        # Save file to upload directory
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(content)
            
        # Parse document with Docling & HybridChunker (with fallback)
        chunks = parse_and_chunk_document(file_path)
        if not chunks:
            return {"status": "error", "message": "無法解析此PDF文件內容或文件為空"}
            
        # Calculate BGE-M3 embeddings & save index into persistent ChromaDB
        build_and_save_index(chunks, collection_name=doc_id)
        
        return {"document_id": doc_id, "status": "success", "chunk_count": len(chunks), "cached": False}
    except Exception as e:
        logger.error(f"Upload and indexing error: {e}", exc_info=True)
        return {"status": "error", "message": f"解析過程發生錯誤: {str(e)}"}

@app.post("/chat")
def chat_doc(req: ChatReq):
    doc_id = str(req.document_id).strip() if req.document_id else ""
    question_str = str(req.question).strip() if req.question else ""
    q = question_str.lower()

    if not doc_id:
        return {"answer": "請先上傳 PDF 文件以獲得 Document ID。", "source_chunks": [], "chunks_details": [], "token_usage": {}}

    # Map-Reduce summary handling for Spec Section 5.2
    if any(k in q for k in ["summary", "summarize", "摘要", "總結"]):
        return handle_map_reduce_summary(doc_id)

    try:
        # Load retriever and LLM from RAG module
        retriever = get_retriever(top_k=config.TOP_K, collection_name=doc_id)
        llm = get_llm()
        rag_graph = build_rag_graph(retriever_instance=retriever, llm_instance=llm)

        initial_state = {
            "query": question_str,
            "context": [],
            "answer": "",
            "token_usage": {}
        }

        final_state = rag_graph.invoke(initial_state)

        answer = final_state.get("answer", "無法生成回答。")
        source_chunks = final_state.get("source_chunks", [])
        chunks_details = final_state.get("chunks_details", [])
        token_usage = final_state.get("token_usage", {})

        return {
            "answer": answer,
            "source_chunks": source_chunks,
            "chunks_details": chunks_details,
            "token_usage": token_usage
        }
    except Exception as e:
        logger.error(f"Chat execution error: {e}", exc_info=True)
        return {"answer": f"對話流程發生錯誤: {str(e)}", "source_chunks": [], "chunks_details": [], "token_usage": {}}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

