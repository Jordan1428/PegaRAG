import logging
from typing import Optional, Any
from langgraph.graph import StateGraph, START, END

from src.pipeline.state import RAGState
from src.pipeline.nodes import retrieve_node, generate_node

logger = logging.getLogger(__name__)

def build_rag_graph(retriever_instance: Optional[Any] = None, llm_instance: Optional[Any] = None):
    """
    Builds and compiles the LangGraph RAG StateGraph pipeline.

    Workflow:
    START -> retrieve_node -> generate_node -> END

    Args:
        retriever_instance: Optional custom retriever instance.
        llm_instance: Optional custom LLM instance.

    Returns:
        Compiled LangGraph instance.
    """
    logger.info("Building LangGraph StateGraph pipeline...")
    builder = StateGraph(RAGState)

    # Wrap nodes with optional pre-injected instances
    def _retrieve(state: RAGState):
        return retrieve_node(state, retriever_instance=retriever_instance)

    def _generate(state: RAGState):
        return generate_node(state, llm_instance=llm_instance)

    # Add Nodes
    builder.add_node("retrieve", _retrieve)
    builder.add_node("generate", _generate)

    # Add Edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    # Compile graph
    app = builder.compile()
    logger.info("LangGraph StateGraph pipeline successfully built and compiled.")
    return app
