import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

import gc
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Testing BGE-M3 Memory Optimization...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Target device: {device.upper()}")

try:
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
        device=device,
        model_kwargs={"use_safetensors": True},
        embed_batch_size=4
    )
    print(f"Successfully loaded {config.EMBEDDING_MODEL_NAME} on {device.upper()}!")
except Exception as e:
    print(f"GPU load error: {e}. Fallback to CPU...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"use_safetensors": True},
        embed_batch_size=4
    )
    print("Successfully loaded on CPU!")
