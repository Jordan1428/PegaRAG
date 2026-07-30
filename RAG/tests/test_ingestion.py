import pytest
from pathlib import Path
from src.ingestion.parser import parse_and_chunk_document, ParsedChunk
from src.ingestion.indexer import build_and_save_index, get_retriever
from src import config

def test_parsed_chunk_structure():
    chunk = ParsedChunk(text="Test chunk text", metadata={"source": "test.pdf"})
    assert chunk.text == "Test chunk text"
    assert chunk.metadata["source"] == "test.pdf"

def test_parse_and_chunk_fallback(tmp_path):
    # Test document parsing handling non-existent path
    with pytest.raises(FileNotFoundError):
        parse_and_chunk_document(tmp_path / "non_existent.pdf")

def test_build_and_get_retriever(tmp_path):
    # Create test chunks
    test_chunks = [
        ParsedChunk(text="LightRAG incorporates graph structures into text indexing.", metadata={"id": 1}),
        ParsedChunk(text="GraphRAG requires traversing communities causing high API cost.", metadata={"id": 2}),
    ]
    
    # Build index with test collection
    test_collection = "test_collection"
    index = build_and_save_index(test_chunks, collection_name=test_collection)
    assert index is not None

    retriever = get_retriever(top_k=2, collection_name=test_collection)
    results = retriever.retrieve("LightRAG")
    assert len(results) > 0
