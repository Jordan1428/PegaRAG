import logging
from typing import Dict, Any, Optional

from src import config
from src.pipeline.state import RAGState
from src.prompts.qa_prompt import QA_PROMPT
from src.llm_factory import get_llm
from src.ingestion.indexer import get_retriever

logger = logging.getLogger(__name__)

def retrieve_node(state: RAGState, retriever_instance: Optional[Any] = None) -> Dict[str, Any]:
    """
    LangGraph Node: retrieve_node
    Retrieves Top-K relevant chunks for the given query from ChromaDB.

    Args:
        state: Current RAG state containing 'query'.
        retriever_instance: Optional pre-initialized LlamaIndex retriever.

    Returns:
        Dict update for state: {"context": List[str]}
    """
    query = state.get("query", "")
    logger.info(f"[retrieve_node] Processing query: '{query}'")

    try:
        retriever = retriever_instance or get_retriever(top_k=config.TOP_K)
        retrieved_nodes = retriever.retrieve(query)
        
        context_list = []
        source_chunks = []
        chunks_details = []

        for idx, node_with_score in enumerate(retrieved_nodes):
            text = node_with_score.node.get_content()
            score = getattr(node_with_score, "score", None)
            score_str = f" (Score: {score:.4f})" if score is not None else ""
            context_list.append(f"[Chunk {idx+1}{score_str}]\n{text}")

            meta = node_with_score.node.metadata or {}
            chunk_id = meta.get("chunk_id", f"chunk_{idx+1}")
            if isinstance(chunk_id, int):
                chunk_id = f"chunk_{chunk_id:03d}"
            else:
                chunk_id = str(chunk_id)
            page_num = meta.get("page_num", meta.get("page", 1))

            source_chunks.append(chunk_id)
            chunks_details.append({
                "chunk_id": chunk_id,
                "content": text,
                "page_num": page_num,
                "score": score
            })

        logger.info(f"[retrieve_node] Successfully retrieved {len(context_list)} chunks.")
        return {
            "context": context_list,
            "source_chunks": source_chunks,
            "chunks_details": chunks_details
        }

    except Exception as e:
        logger.error(f"[retrieve_node] Error during retrieval: {e}", exc_info=True)
        return {
            "context": [f"Error retrieving context for query: {query}"],
            "source_chunks": [],
            "chunks_details": []
        }


def generate_node(state: RAGState, llm_instance: Optional[Any] = None) -> Dict[str, Any]:
    """
    LangGraph Node: generate_node
    Injects query and retrieved context into Prompt Template and calls LLM.

    Args:
        state: Current RAG state containing 'query' and 'context'.
        llm_instance: Optional pre-initialized BaseChatModel instance.

    Returns:
        Dict update for state: {"answer": str}
    """
    query = state.get("query", "")
    context_list = state.get("context", [])
    
    formatted_context = "\n\n".join(context_list) if context_list else "No relevant context found."
    logger.info(f"[generate_node] Generating answer for query: '{query}' with {len(context_list)} context chunks.")

    # Format Prompt
    prompt_str = QA_PROMPT.format(context=formatted_context, query=query)

    try:
        llm = llm_instance or get_llm()
        response = llm.invoke(prompt_str)
        raw_content = response.content if hasattr(response, "content") else str(response)
        
        import ast
        if isinstance(raw_content, list):
            answer = "".join(item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in raw_content)
        elif isinstance(raw_content, str) and raw_content.startswith("[{'") and "'text':" in raw_content:
            try:
                parsed_list = ast.literal_eval(raw_content)
                answer = "".join(item.get("text", "") for item in parsed_list if isinstance(item, dict))
            except Exception:
                answer = raw_content
        else:
            answer = str(raw_content)

        # Extract token usage metadata from response
        token_usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            raw_meta = response.usage_metadata
            token_usage = {
                "input_tokens": raw_meta.get("input_tokens", 0),
                "output_tokens": raw_meta.get("output_tokens", 0),
                "total_tokens": raw_meta.get("total_tokens", 0)
            }
        elif hasattr(response, "response_metadata") and response.response_metadata:
            token_usage = response.response_metadata.get("token_usage", {}) or response.response_metadata.get("usage", {})

        logger.info(f"[generate_node] Answer generated successfully. Token usage: {token_usage}")
        return {"answer": answer, "token_usage": token_usage}

    except Exception as e:
        logger.error(f"[generate_node] Error generating answer: {e}", exc_info=True)
        return {"answer": f"Error generating answer: {e}", "token_usage": {}}
