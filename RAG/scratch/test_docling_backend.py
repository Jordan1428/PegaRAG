import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

pdf_path = Path("data/raw/2410.05779v3.pdf")

options = PdfPipelineOptions()
options.do_ocr = False
options.do_table_structure = False

converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(
            pipeline_options=options,
            backend=PyPdfiumDocumentBackend
        )
    }
)

result = converter.convert(str(pdf_path))
docling_doc = result.document
md = docling_doc.export_to_markdown()

tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
chunker = HybridChunker(tokenizer=tokenizer, max_tokens=config.CHUNK_SIZE, merge_peers=True, repeat_table_header=True)

chunks = list(chunker.chunk(docling_doc))
print(f"Docling with PyPdfiumBackend: Pages={len(docling_doc.pages)}, Markdown length={len(md)}, Chunks={len(chunks)}")
print("Last 150 chars of markdown:\n", md[-150:].replace("\n", " "))
