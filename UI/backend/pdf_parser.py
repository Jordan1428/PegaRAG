import io, logging
from typing import List, Dict, Any
logger = logging.getLogger(__name__)

def table_to_md(tbl):
    if not tbl: return ""
    clean = [[str(c).replace("\n"," ").strip() if c is not None else "" for c in r] for r in tbl if any(r)]
    clean = [r for r in clean if any(r)]
    if not clean: return ""
    cols = max(len(r) for r in clean)
    hdr = clean[0] + [f"Col_{i+1}" for i in range(cols-len(clean[0]))]
    res = ["| " + " | ".join(hdr) + " |", "| " + " | ".join(["---"]*cols) + " |"]
    for r in clean[1:]:
        res.append("| " + " | ".join((r + [""]*(cols-len(r)))[:cols]) + " |")
    return "\n".join(res)

def parse_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    pages = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, p in enumerate(pdf.pages, 1):
                txt = p.extract_text() or ""
                tbls = [table_to_md(t) for t in (p.extract_tables() or []) if t]
                tbls = [t for t in tbls if t]
                ftxt = txt + ("\n\n### Tables:\n" + "\n\n".join(tbls) if tbls else "")
                pages.append({"page_num": i, "text": ftxt.strip(), "has_table": len(tbls)>0})
        if pages and any(p["text"] for p in pages): return pages
    except Exception as e: logger.warning(f"pdfplumber err: {e}")

    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        p = doc[i]
        txt = p.get_text()
        tbls = []
        try:
            tabs = p.find_tables()
            if tabs and tabs.tables:
                tbls = [table_to_md(t.extract()) for t in tabs.tables if t]
                tbls = [t for t in tbls if t]
        except Exception: pass
        ftxt = txt + ("\n\n### Tables:\n" + "\n\n".join(tbls) if tbls else "")
        pages.append({"page_num": i+1, "text": ftxt.strip(), "has_table": len(tbls)>0})
    return pages
