import pytest
from src.pipeline.state import RAGState
from src.pipeline.nodes import retrieve_node, generate_node
from src.pipeline.graph import build_rag_graph
from src.llm_factory import get_llm, MockChatModel

def test_llm_factory_mock():
    llm = get_llm(llm_type="mock")
    assert isinstance(llm, MockChatModel)
    response = llm.invoke("Summarize this document.")
    assert "LightRAG" in response.content

def test_rag_nodes_and_graph():
    # Mock LLM and Graph test
    mock_llm = MockChatModel()
    app = build_rag_graph(llm_instance=mock_llm)
    
    initial_state: RAGState = {"query": "What is LightRAG?", "context": [], "answer": ""}
    final_state = app.invoke(initial_state)

    assert "query" in final_state
    assert "context" in final_state
    assert "answer" in final_state
    assert len(final_state["answer"]) > 0
