import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.parser import parse_and_chunk_document

pdf_path = Path("data/raw/2410.05779v3.pdf")
print(f"Parsing PDF: {pdf_path}")

chunks = parse_and_chunk_document(pdf_path)
print(f"Total chunks returned by parser: {len(chunks)}")

for i, c in enumerate(chunks):
    first_line = c.text.strip().split("\n")[0] if c.text else ""
    print(f"Chunk #{i+1} (len={len(c.text)}): {first_line[:100]}")
