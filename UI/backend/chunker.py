from typing import List, Dict, Any

def chunk_document_pages(pages: List[Dict[str, Any]], max_chars: int = 1400, overlap: int = 150) -> List[Dict[str, Any]]:
    chunks, idx = [], 0
    for p in pages:
        txt = p.get("text", "").strip()
        if not txt: continue
        i = 0
        while i < len(txt):
            end = min(i + max_chars, len(txt))
            sub = txt[i:end].strip()
            if sub:
                idx += 1
                chunks.append({"chunk_id": f"chunk_{idx:03d}", "text": sub, "page_num": p.get("page_num", 1), "chunk_index": idx, "has_table": p.get("has_table", False) or ("|" in sub)})
            if end >= len(txt): break
            i = max(end - overlap, i + 1)
    return chunks
