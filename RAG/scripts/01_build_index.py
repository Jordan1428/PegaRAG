import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.ingestion.parser import parse_and_chunk_document
from src.ingestion.indexer import build_and_save_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("Step 1: Building RAG Index from Raw PDF Documents")
    print("=" * 60)

    # Search for PDF files in data/raw
    pdf_files = list(config.RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {config.RAW_DATA_DIR}. Please place '2410.05779v3.pdf' or target PDF in data/raw/")
        sys.exit(1)

    target_pdf = pdf_files[0]
    logger.info(f"Target PDF file identified: {target_pdf}")

    # Step 1: Parse document with Docling & HybridChunker
    logger.info("Parsing document with Docling & HybridChunker...")
    chunks = parse_and_chunk_document(target_pdf)
    logger.info(f"Extracted {len(chunks)} structured chunks.")

    # Step 2: Calculate BGE-M3 Embeddings and Store in ChromaDB
    logger.info(f"Indexing chunks with BGE-M3 into persistent ChromaDB at {config.CHROMADB_DIR}...")
    index = build_and_save_index(chunks)

    print("\n" + "=" * 60)
    print("✅ Indexing process completed successfully!")
    print(f"ChromaDB Vector Store Path: {config.CHROMADB_DIR}")
    print(f"Total Chunks Indexed: {len(chunks)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
