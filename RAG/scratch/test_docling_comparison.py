import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice

pdf_path = Path("data/raw/2410.05779v3.pdf")

# Test 1: do_table_structure = True
opt1 = PdfPipelineOptions()
opt1.do_ocr = False
opt1.do_table_structure = True
opt1.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CUDA)

conv1 = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opt1)})
res1 = conv1.convert(str(pdf_path))
md1 = res1.document.export_to_markdown()

print(f"Test 1 (do_table_structure=True): Pages={len(res1.document.pages)}, Markdown chars={len(md1)}")
print(f"Test 1 last 200 chars:\n{md1[-200:]}\n")

# Test 2: do_table_structure = False
opt2 = PdfPipelineOptions()
opt2.do_ocr = False
opt2.do_table_structure = False
opt2.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CUDA)

conv2 = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opt2)})
res2 = conv2.convert(str(pdf_path))
md2 = res2.document.export_to_markdown()

print(f"Test 2 (do_table_structure=False): Pages={len(res2.document.pages)}, Markdown chars={len(md2)}")
print(f"Test 2 last 200 chars:\n{md2[-200:]}\n")
