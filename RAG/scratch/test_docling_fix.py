import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

import logging
logging.basicConfig(level=logging.INFO)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

pdf_path = Path("data/raw/2410.05779v3.pdf")

options = PdfPipelineOptions()
options.do_ocr = False
options.do_table_structure = False  # Disable heavy table structure vision model to prevent std::bad_alloc
options.do_picture_classification = False
options.generate_page_images = False
options.generate_picture_images = False
options.generate_table_images = False

print("Testing Docling with do_table_structure=False...")
converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=options)
    }
)

result = converter.convert(str(pdf_path))
docling_doc = result.document
print(f"Docling successfully parsed document with {len(docling_doc.pages)} pages.")

tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=config.CHUNK_SIZE,
    merge_peers=True,
    repeat_table_header=True,
)

doc_chunks = list(chunker.chunk(docling_doc))
print(f"Docling HybridChunker produced {len(doc_chunks)} chunks.")

for idx, chunk in enumerate(doc_chunks):
    first_line = chunk.text.strip().split("\n")[0] if chunk.text else ""
    print(f"Chunk #{idx+1} (len={len(chunk.text)}): {first_line[:80]}")

print("\nLast chunk full text preview:")
print(doc_chunks[-1].text[-300:])
