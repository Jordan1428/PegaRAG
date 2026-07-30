import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.pipeline.graph import build_rag_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_query(query: str):
    """Executes a query through the LangGraph RAG pipeline and prints results."""
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    rag_graph = build_rag_graph()
    initial_state = {"query": query, "context": [], "answer": "", "token_usage": {}}

    logger.info("Executing LangGraph pipeline...")
    final_state = rag_graph.invoke(initial_state)

    print("\n--- [Retrieved Context Chunks] ---")
    for idx, ctx in enumerate(final_state.get("context", [])):
        print(f"\n[Chunk #{idx + 1}]")
        print(ctx[:300] + ("..." if len(ctx) > 300 else ""))

    print("\n--- [Final Generated Answer] ---")
    print(final_state.get("answer", "No answer generated."))

    token_usage = final_state.get("token_usage", {})
    if token_usage:
        print("\n--- [Token Usage Statistics] ---")
        in_tok = token_usage.get("input_tokens", token_usage.get("prompt_tokens", "N/A"))
        out_tok = token_usage.get("output_tokens", token_usage.get("completion_tokens", "N/A"))
        tot_tok = token_usage.get("total_tokens", "N/A")
        print(f"  • Prompt (Input) Tokens: {in_tok}")
        print(f"  • Completion (Output) Tokens: {out_tok}")
        print(f"  • Total Tokens: {tot_tok}")

    print("=" * 60 + "\n")
    return final_state

def main():
    parser = argparse.ArgumentParser(description="Run single query against LangGraph RAG pipeline.")
    parser.add_argument("--query", "-q", type=str, help="Query string to answer.")
    args = parser.parse_args()

    if args.query:
        run_query(args.query)
    else:
        # Interactive CLI mode
        print("Starting Interactive RAG Query CLI (type 'exit' or 'quit' to stop)...")
        while True:
            try:
                user_input = input("\nEnter Question: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                if user_input:
                    run_query(user_input)
            except (KeyboardInterrupt, EOFError):
                break
        print("Exiting CLI.")

if __name__ == "__main__":
    main()
