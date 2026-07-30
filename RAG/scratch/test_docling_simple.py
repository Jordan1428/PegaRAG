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

# Let's inspect PdfPipelineOptions attributes
options = PdfPipelineOptions()
print("PdfPipelineOptions fields:", options.__dict__.keys())

# Let's test disabling layout model if possible, or using SimplePipeline
try:
    from docling.pipeline.simple_pipeline import SimplePipeline
    print("Testing SimplePipeline...")
    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_cls=SimplePipeline)
        }
    )
    result = converter.convert(str(pdf_path))
    docling_doc = result.document
    print(f"SimplePipeline parsed document with {len(docling_doc.pages)} pages.")
    markdown_text = docling_doc.export_to_markdown()
    print(f"Total exported Markdown length: {len(markdown_text)} chars.")
    print("Last 300 chars:")
    print(markdown_text[-300:])
except Exception as e:
    print(f"SimplePipeline test error: {e}")
