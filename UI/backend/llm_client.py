import requests
from typing import List, Dict, Any

class LLMClient:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3"):
        self.ollama_url = ollama_url
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI document assistant.") -> str:
        # Enforce strict 10k token limit (~30,000 characters)
        if len(prompt) > 30000:
            prompt = prompt[:30000] + "\n...[Context truncated under 10k tokens limit]..."
        
        try:
            r = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False
                },
                timeout=60
            )
            if r.status_code == 200:
                resp = r.json().get("response", "").strip()
                if resp:
                    return resp
        except Exception as e:
            print(f"[Ollama LLM Warning] Could not reach Ollama at {self.ollama_url}: {e}")
        
        return self._fallback_synthesis(prompt)

    def _fallback_synthesis(self, prompt: str) -> str:
        return (
            "⚠️ **[Note: Local Ollama LLM service is offline or loading]**\n"
            "*(If you have Ollama installed, start it using `ollama run llama3`)*\n\n"
            "**[Extracted Relevant Document Chunks]**\n\n"
            f"{prompt[:3000]}\n..."
        )

    def map_reduce_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Map-Reduce strategy for summarizing documents exceeding context size (Spec section 5.2).
        """
        if not chunks:
            return {"answer": "未找到文件內容可供摘要。", "source_chunks": []}

        # Step 1: Map stage - select key chunks across the document & extract summaries
        step = max(1, len(chunks) // 8)
        selected_chunks = chunks[::step][:8]
        
        map_summaries = []
        source_ids = []

        for c in selected_chunks:
            cid = c.get("chunk_id", "chunk")
            content = c.get("content", c.get("text", ""))
            source_ids.append(cid)
            map_prompt = f"Summarize this text section concisely:\n{content[:1500]}"
            summary_text = self.generate(map_prompt, system_prompt="Summarize the key information of the given chunk.")
            map_summaries.append(f"• [{cid}]: {summary_text}")

        # Step 2: Reduce stage - synthesize into final overall summary
        combined = "\n\n".join(map_summaries)
        reduce_prompt = (
            "Below are section summaries of the document. "
            "Synthesize them into a comprehensive, structured overall document summary highlighting the main contribution, methodology, and key results:\n\n"
            f"{combined}"
        )
        
        final_answer = self.generate(reduce_prompt, system_prompt="You are an expert academic paper reviewer.")
        
        # 回傳包含詳細切塊內容的字典，供前端檢索區塊展示
        chunks_details = [
            {
                "chunk_id": c.get("chunk_id", "chunk"),
                "content": c.get("content", c.get("text", "")),
                "page_num": c.get("page_num", 1)
            }
            for c in selected_chunks
        ]
        return {"answer": final_answer, "source_chunks": source_ids, "chunks_details": chunks_details}
