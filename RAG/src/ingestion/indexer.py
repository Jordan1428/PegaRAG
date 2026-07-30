import os
# Ensure TensorFlow is disabled BEFORE any HuggingFace/LlamaIndex imports
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from src import config

import gc
import logging
from typing import List, Optional
import chromadb
import torch

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.ingestion.parser import ParsedChunk

logger = logging.getLogger(__name__)

def initialize_embeddings():
    """Configures LlamaIndex global embedding model to BGE-M3 with FP16 GPU memory optimization."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing Embedding Model '{config.EMBEDDING_MODEL_NAME}' on device: '{device.upper()}' (FP16 Mode)")

    try:
        # Use FP16 (torch.float16) to cut VRAM/RAM usage in half (from 2.3GB down to 1.1GB)
        embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
            embed_batch_size=4
        )
    except Exception as e:
        logger.warning(f"GPU FP16 load warning ({e}). Falling back to standard load.")
        embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=device,
            embed_batch_size=4
        )

    Settings.embed_model = embed_model
    return embed_model


def build_and_save_index(chunks: List[ParsedChunk], collection_name: Optional[str] = None) -> VectorStoreIndex:
    """
    Converts ParsedChunks into LlamaIndex TextNodes, computes embeddings,
    and stores them into local persistent ChromaDB.

    Args:
        chunks: List of ParsedChunk objects.
        collection_name: Name of ChromaDB collection.

    Returns:
        VectorStoreIndex: Persistent LlamaIndex vector store index instance.
    """
    collection_name = collection_name or config.COLLECTION_NAME
    embed_model = initialize_embeddings()

    # Convert ParsedChunks into LlamaIndex TextNodes
    nodes = []
    for idx, chunk in enumerate(chunks):
        node = TextNode(
            text=chunk.text,
            id_=f"node_{idx}",
            metadata=chunk.metadata
        )
        nodes.append(node)

    logger.info(f"Converted {len(nodes)} chunks into LlamaIndex TextNodes.")

    # Connect to local ChromaDB client
    db_path = str(config.CHROMADB_DIR)
    logger.info(f"Connecting to ChromaDB at: {db_path}")
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    # Create ChromaVectorStore & StorageContext
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build VectorStoreIndex
    logger.info("Building VectorStoreIndex in ChromaDB...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    logger.info("Index built and saved successfully to ChromaDB.")
    return index


def get_retriever(top_k: Optional[int] = None, collection_name: Optional[str] = None):
    """
    Loads persistent ChromaDB vector store and returns a LlamaIndex Retriever.

    Args:
        top_k: Number of Top-K results to retrieve (default: config.TOP_K).
        collection_name: Name of ChromaDB collection.

    Returns:
        Retriever instance.
    """
    top_k = top_k or config.TOP_K
    collection_name = collection_name or config.COLLECTION_NAME
    embed_model = initialize_embeddings()

    db_path = str(config.CHROMADB_DIR)
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    return index.as_retriever(similarity_top_k=top_k)
