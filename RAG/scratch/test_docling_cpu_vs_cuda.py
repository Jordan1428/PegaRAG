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

# Test CPU Accelerator Options
opt_cpu = PdfPipelineOptions()
opt_cpu.do_ocr = False
opt_cpu.do_table_structure = False
opt_cpu.do_picture_classification = False
opt_cpu.generate_page_images = False
opt_cpu.generate_picture_images = False
opt_cpu.generate_table_images = False
opt_cpu.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)

conv_cpu = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opt_cpu)})
res_cpu = conv_cpu.convert(str(pdf_path))
doc_cpu = res_cpu.document

tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
chunker = HybridChunker(tokenizer=tokenizer, max_tokens=config.CHUNK_SIZE, merge_peers=True, repeat_table_header=True)

chunks_cpu = list(chunker.chunk(doc_cpu))
print(f"Docling CPU mode: Pages={len(doc_cpu.pages)}, Markdown length={len(doc_cpu.export_to_markdown())}, Chunks={len(chunks_cpu)}")
print("Last line of CPU markdown:", doc_cpu.export_to_markdown()[-150:].replace("\n", " "))
