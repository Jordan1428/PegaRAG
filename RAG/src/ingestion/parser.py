import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import gc
import torch

from src import config

logger = logging.getLogger(__name__)

class ParsedChunk:
    """Represents a chunk extracted by Docling & HybridChunker."""
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self):
        return f"<ParsedChunk length={len(self.text)} meta={self.metadata}>"


def parse_and_chunk_document(file_path: Union[str, Path]) -> List[ParsedChunk]:
    """
    Parses a PDF document using Docling DocumentConverter on CPU with PyPdfiumDocumentBackend.
    Running Docling on CPU leaves 100% of GTX 1650 GPU VRAM (4GB) dedicated for BGE-M3 embedding.

    Args:
        file_path: Path to the raw PDF document.

    Returns:
        List[ParsedChunk]: Chunks with extracted text and structured metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    logger.info(f"Parsing document with Docling (CPU Mode for VRAM protection): {file_path}")
    chunks: List[ParsedChunk] = []

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.chunking import HybridChunker
        from transformers import AutoTokenizer

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        # Dedicate CPU for Docling layout parsing so GPU VRAM is completely free for BGE-M3
        pipeline_options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)

        converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        result = converter.convert(str(file_path))
        docling_doc = result.document

        # Setup Tokenizer-aware HybridChunker for BGE-M3
        logger.info(f"Initializing HybridChunker with tokenizer '{config.EMBEDDING_MODEL_NAME}'")
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.warning(f"Could not load AutoTokenizer for {config.EMBEDDING_MODEL_NAME}: {e}. Falling back to default tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=config.CHUNK_SIZE,
            merge_peers=True,
            repeat_table_header=True,
        )

        doc_chunks = list(chunker.chunk(docling_doc))
        logger.info(f"Docling HybridChunker produced {len(doc_chunks)} chunks covering {len(docling_doc.pages)} pages.")

        for idx, chunk in enumerate(doc_chunks):
            chunk_text = chunker.serialize(chunk) if hasattr(chunker, "serialize") else str(chunk.text)
            
            # Extract page_num from Docling provenance metadata
            page_num = 1
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        for p in item.prov:
                            if hasattr(p, "page_no") and p.page_no:
                                page_num = p.page_no
                                break
                    if page_num > 1:
                        break

            cid = f"chunk_{idx+1:03d}"
            meta = {
                "source": file_path.name,
                "chunk_id": cid,
                "page_num": page_num
            }

            chunks.append(ParsedChunk(text=chunk_text, metadata=meta))

    except Exception as e:
        logger.error(f"Docling parsing error: {e}. Falling back to PDFium extraction.", exc_info=True)
        chunks = _pdfium_fallback_parse(file_path)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Final total chunks generated: {len(chunks)}")
    return chunks


def _pdfium_fallback_parse(file_path: Path) -> List[ParsedChunk]:
    """Fallback parser using pypdfium2."""
    chunks = []
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(file_path))
        total_pages = len(pdf)

        page_texts = []
        for page_idx in range(total_pages):
            text = pdf[page_idx].get_textpage().get_text_range().strip()
            if text:
                page_texts.append((page_idx + 1, text))

        chunk_size_chars = config.CHUNK_SIZE * 4
        overlap_chars = config.CHUNK_OVERLAP * 4
        current_text = ""
        current_pages = []
        counter = 0

        for p_num, p_text in page_texts:
            for paragraph in p_text.split("\n\n"):
                p_clean = paragraph.strip()
                if not p_clean:
                    continue
                if len(current_text) + len(p_clean) > chunk_size_chars and current_text:
                    cid = f"chunk_{counter+1:03d}"
                    page_val = current_pages[0] if current_pages else 1
                    chunks.append(ParsedChunk(text=current_text, metadata={"source": file_path.name, "chunk_id": cid, "page_num": page_val}))
                    counter += 1
                    current_text = current_text[-overlap_chars:] + "\n\n" + p_clean
                    current_pages = [p_num]
                else:
                    current_text = current_text + ("\n\n" if current_text else "") + p_clean
                    current_pages.append(p_num)

        if current_text.strip():
            cid = f"chunk_{counter+1:03d}"
            page_val = current_pages[0] if current_pages else 1
            chunks.append(ParsedChunk(text=current_text, metadata={"source": file_path.name, "chunk_id": cid, "page_num": page_val}))

    except Exception as e:
        logger.error(f"Fallback parsing failed: {e}")
    return chunks
