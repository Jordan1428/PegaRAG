import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

pdf_path = Path("data/raw/2410.05779v3.pdf")

options = PdfPipelineOptions()
options.do_ocr = False
options.do_table_structure = False
options.do_picture_classification = False
options.generate_page_images = False
options.generate_picture_images = False
options.generate_table_images = False
options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CUDA)

converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=options)
    }
)

result = converter.convert(str(pdf_path))
docling_doc = result.document
md = docling_doc.export_to_markdown()

print(f"Docling parsed pages: {len(docling_doc.pages)}")
print(f"Docling exported markdown total length: {len(md)} characters.")

tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=config.CHUNK_SIZE,
    merge_peers=False,  # Test without merge_peers
    repeat_table_header=True,
)

doc_chunks = list(chunker.chunk(docling_doc))
print(f"HybridChunker (merge_peers=False) produced: {len(doc_chunks)} chunks.")

total_chunk_chars = sum([len(c.text) for c in doc_chunks])
print(f"Total characters in all chunks: {total_chunk_chars}")

for idx, c in enumerate(doc_chunks):
    chunk_str = chunker.serialize(c) if hasattr(chunker, "serialize") else str(c.text)
    first_line = chunk_str.strip().split("\n")[0] if chunk_str else ""
    print(f"Chunk #{idx+1} ({len(chunk_str)} chars): {first_line[:80]}")
