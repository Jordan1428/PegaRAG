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
import torch

pdf_path = Path("data/raw/2410.05779v3.pdf")

options = PdfPipelineOptions()
options.do_ocr = False
options.do_table_structure = False
options.do_picture_classification = False
options.generate_page_images = False
options.generate_picture_images = False
options.generate_table_images = False
options.force_backend_text = True  # Direct text stream backend without high-DPI C++ page rasterization

print("Testing Docling with force_backend_text=True...")
converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=options)
    }
)

result = converter.convert(str(pdf_path))
docling_doc = result.document
print(f"Docling parsed {len(docling_doc.pages)} pages cleanly.")
markdown_text = docling_doc.export_to_markdown()
print(f"Markdown length: {len(markdown_text)} chars.")
