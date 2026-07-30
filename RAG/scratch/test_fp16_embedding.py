import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

import gc
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Testing FP16 BGE-M3 on CUDA...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

embed_model = HuggingFaceEmbedding(
    model_name=config.EMBEDDING_MODEL_NAME,
    trust_remote_code=True,
    device="cuda",
    model_kwargs={"torch_dtype": torch.float16},
    embed_batch_size=4
)

print(f"FP16 BGE-M3 loaded successfully on CUDA! Test embedding 1 text...")
vec = embed_model.get_text_embedding("Test RAG embedding on GPU")
print(f"Vector dimension: {len(vec)}")
