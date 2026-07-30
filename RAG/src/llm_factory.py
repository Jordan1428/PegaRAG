import os
import logging
from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from src import config

logger = logging.getLogger(__name__)

class MockChatModel(BaseChatModel):
    """
    A lightweight Mock Chat Model for offline testing or when no API keys/Ollama daemons are available.
    Extracts key information from context if available in prompt, or returns a structured response.
    """
    model_name: str = "mock-llm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_msg = messages[-1].content if messages else ""
        
        # Simple extraction heuristics for offline mock testing
        text = f"[Mock LLM Response]\nBased on context, here is the answer to your query: '{last_msg[-100:]}'."
        if "Summarize" in last_msg or "summary" in last_msg.lower():
            text = (
                "LightRAG is a RAG system combining graph-based text indexing with a dual-level retrieval framework, "
                "addressing comprehensive retrieval, retrieval efficiency, and fast adaptation to new data."
            )
        elif "Compare" in last_msg or "GraphRAG" in last_msg:
            text = (
                "LightRAG outperforms GraphRAG across multiple datasets. GraphRAG requires traversing communities causing "
                "many API calls, while LightRAG uses dual-level retrieval with single API calls per query and incremental updates."
            )

        gen = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[gen])

    @property
    def _llm_type(self) -> str:
        return "mock"


def get_llm(llm_type: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.0) -> BaseChatModel:
    """
    Factory function to instantiate and return a BaseChatModel instance
    based on system configuration or provided parameters.

    Args:
        llm_type: Optional override for LLM type ('openai', 'ollama', 'gemini', 'anthropic', 'mock').
        model_name: Optional override for model name.
        temperature: Sampling temperature for generation.

    Returns:
        BaseChatModel: LangChain compatible chat model.
    """
    llm_type = (llm_type or config.LLM_TYPE).lower()
    model_name = model_name or config.LLM_MODEL_NAME

    logger.info(f"Instantiating LLM Factory with type='{llm_type}', model='{model_name}'")

    if llm_type == "openai":
        from langchain_openai import ChatOpenAI
        api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        api_base = config.OPENAI_API_BASE or os.getenv("OPENAI_API_BASE")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Falling back to MockChatModel.")
            return MockChatModel()
        
        kwargs = {
            "model": model_name or "gpt-4o-mini",
            "openai_api_key": api_key,
            "temperature": temperature
        }
        if api_base:
            kwargs["openai_api_base"] = api_base
            
        return ChatOpenAI(**kwargs)

    elif llm_type == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=model_name or "llama3",
                base_url=config.OLLAMA_BASE_URL,
                temperature=temperature
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ChatOllama ({e}). Falling back to MockChatModel.")
            return MockChatModel()

    elif llm_type in ["gemini", "google"]:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY / GOOGLE_API_KEY not set. Falling back to MockChatModel.")
                return MockChatModel()
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-1.5-flash",
                google_api_key=api_key,
                temperature=temperature
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGoogleGenerativeAI ({e}). Falling back to MockChatModel.")
            return MockChatModel()

    elif llm_type == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            api_key = config.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set. Falling back to MockChatModel.")
                return MockChatModel()
            return ChatAnthropic(
                model_name=model_name or "claude-3-5-sonnet-20241022",
                anthropic_api_key=api_key,
                temperature=temperature
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ChatAnthropic ({e}). Falling back to MockChatModel.")
            return MockChatModel()

    elif llm_type == "mock":
        return MockChatModel()

    else:
        logger.warning(f"Unknown LLM_TYPE '{llm_type}'. Returning MockChatModel.")
        return MockChatModel()
