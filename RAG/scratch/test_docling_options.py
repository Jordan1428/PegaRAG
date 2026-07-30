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

pdf_path = Path("data/raw/2410.05779v3.pdf")

options = PdfPipelineOptions()
options.do_ocr = False
options.do_picture_classification = False
options.generate_page_images = False
options.generate_picture_images = False
options.generate_table_images = False

print("Testing Docling with disabled images/OCR...")
converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=options)
    }
)

result = converter.convert(str(pdf_path))
doc = result.document
print(f"Docling produced document with {len(doc.pages)} pages.")

markdown_text = doc.export_to_markdown()
print(f"Total exported Markdown length: {len(markdown_text)} characters.")
print("Last 300 chars of Markdown:")
print(markdown_text[-300:])
