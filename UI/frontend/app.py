import os
import requests
import gradio as gr

BACKEND_URL = "http://127.0.0.1:8000"

RAG_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RAG", "data", "raw"))
DEFAULT_PAPER_PATH = os.path.join(RAG_DATA_DIR, "2410.05779v3.pdf")

def upload_file_to_backend(file_obj, force_reparse=False):
    if not file_obj:
        return "⚠️ 請先選擇要上傳的 PDF 檔案！", "", "未選擇檔案"
    try:
        file_path = file_obj.name
        if not os.path.exists(file_path):
            return f"❌ 找不到檔案: {file_path}", "", "檔案不存在"
            
        import time
        resp = None
        for attempt in range(5):
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (os.path.basename(file_path), f, "application/pdf")}
                    req_data = {"force_reparse": "true" if force_reparse else "false"}
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files, data=req_data, timeout=600)
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt < 4:
                    time.sleep(1.5)
                else:
                    raise
        data = resp.json() if resp else {}
        if data.get("status") == "success":
            doc_id = data.get("document_id")
            chunk_cnt = data.get("chunk_count")
            cached = data.get("cached", False)
            msg = "⚡ 載入已解析快取成功！" if cached else ("🔄 強制重新解析與建庫完成！" if force_reparse else "✅ 上傳並解析成功！")
            return msg, doc_id, f"總 Chunk 數: {chunk_cnt} {'(使用向量庫快取)' if cached else '(已重新提取頁碼與向量)'}"
        else:
            return f"❌ 上傳失敗: {data.get('message')}", "", "解析失敗"
    except Exception as e:
        return f"❌ 連線錯誤: {str(e)}", "", "無法連接後端 (Port 8000)"

def load_default_paper(force_reparse=False):
    if not os.path.exists(DEFAULT_PAPER_PATH):
        return f"❌ 找不到預設論文檔案 ({DEFAULT_PAPER_PATH})", "", "檔案不存在"
    class DummyFile:
        def __init__(self, path):
            self.name = path
    return upload_file_to_backend(DummyFile(DEFAULT_PAPER_PATH), force_reparse=force_reparse)

import re

def clean_text_newlines(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # Compress 3 or more consecutive newlines into double newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# 註解 [修改 1]: 修改 chat_fn，除了產生 LLM 回答之外，同步格式化並回傳檢索出來的 Chunk 內文
def chat_fn(message, history, doc_id):
    doc_id_str = str(doc_id).strip() if doc_id is not None else ""
    message_str = str(message).strip() if message is not None else ""

    if not doc_id_str:
        return "⚠️ 請先在上側或左側上傳 PDF 文件（或點擊載入預設 LightRAG 論文）！", "*(尚無檢索內容)*"
    try:
        payload = {"document_id": doc_id_str, "question": message_str}
        resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=300)
        data = resp.json()
        
        ans_raw = data.get("answer", "無回應")
        
        # 針對 [{'text': '...', 'type': 'text'}] 格式精準提取
        if isinstance(ans_raw, list):
            # 走訪列表，直接把字典裡的 'text' 抓出來合併
            ans = "".join(item.get("text", "") for item in ans_raw if isinstance(item, dict))
        else:
            ans = str(ans_raw)
            
        ans = ans.strip()

        sources = data.get("source_chunks", [])
        chunks_details = data.get("chunks_details", [])
        token_usage = data.get("token_usage", {})

        if sources:
            clean_sources = [str(s).strip() for s in sources if str(s).strip()]
            if clean_sources:
                ans += f"\n\n📍 **參考片段:** {', '.join(clean_sources)}"

        inp_t = token_usage.get("input_tokens", token_usage.get("prompt_tokens", 0))
        out_t = token_usage.get("output_tokens", token_usage.get("completion_tokens", 0))
        tot_t = token_usage.get("total_tokens", inp_t + out_t)

        if tot_t > 0:
            ans += f"\n\n📊 **Token 消耗:** 輸入 `{inp_t:,}` | 輸出 `{out_t:,}` | 總耗用 `{tot_t:,}`"
            
        # 格式化檢索出來的 Chunk 內容展示文字
        if chunks_details:
            formatted_chunks = []
            for item in chunks_details:
                cid = str(item.get("chunk_id", "Chunk"))
                page = item.get("page_num", item.get("page", 1))
                text = clean_text_newlines(str(item.get("content", item.get("text", ""))))
                formatted_chunks.append(f"#### 📌 [{cid}] (第 {page} 頁)\n{text}")
            chunks_display = "\n\n---\n\n".join(formatted_chunks)
        else:
            chunks_display = "*(此問題未檢索到相關 Chunk)*"

        return ans, chunks_display
    except Exception as e:
        return f"❌ 查詢失敗 (請確保後端伺服器運行中): {str(e)}", f"❌ 錯誤: {str(e)}"

# Custom High-Contrast Theme with CSS Variables & Strict Component Overrides
custom_css = """
:root, body, .gradio-container {
    --body-text-color: #ffffff !important;
    --body-text-color-subdued: #f1f5f9 !important;
    --background-fill-primary: #111827 !important;
    --background-fill-secondary: #1f2937 !important;
    --block-background-fill: #111827 !important;
    --block-border-color: #374151 !important;
    --block-label-text-color: #ffffff !important;
    --block-title-text-color: #ffffff !important;
    --input-background-fill: #1f2937 !important;
    --input-border-color: #4b5563 !important;
    --input-text-color: #ffffff !important;
    --input-placeholder-color: #9ca3af !important;
    --table-text-color: #ffffff !important;
    --table-border-color: #374151 !important;
    
    background-color: #0b0f19 !important;
    color: #ffffff !important;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    transition: background-color 0.3s ease, color 0.3s ease;
}

/* Hide Footer (Use via API) */
footer, .footer, footer.svelte-152q79, footer.svelte-11md07n, footer.svelte-mp72ic, div[class*="footer"] {
    display: none !important;
}

/* Dashboard Banner (No Box Border) */
.hero-banner {
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
    margin-bottom: 8px !important;
    box-shadow: none !important;
}

.hero-title {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #c7d2fe 0%, #818cf8 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
}

.hero-subtitle {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
    margin-top: 2px !important;
}

/* Glassmorphic Panels */
.glass-panel {
    background: rgba(17, 24, 39, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border: 1.5px solid #374151 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4) !important;
}

/* All Headings & Labels in Dark Mode */
.glass-panel label, .glass-panel label *, .block-label, label span, .label-text, .form-label, label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

h1, h2, h3, h4, h5, h6, .block-title {
    color: #ffffff !important;
    font-weight: 800 !important;
}

p, span, div, caption {
    color: #ffffff !important;
}

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}

.btn-preset {
    background: #1f2937 !important;
    border: 1.5px solid #4b5563 !important;
    color: #f3f4f6 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

.btn-preset:hover {
    background: #374151 !important;
    border-color: #818cf8 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Form Inputs, Textarea, User Input Textbox */
input, textarea, select, 
.gradio-textbox input, .gradio-textbox textarea, 
div[data-testid="textbox"] input, div[data-testid="textbox"] textarea {
    background-color: #1f2937 !important;
    border: 1.5px solid #4b5563 !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
}

/* Comprehensive PDF File Upload Component Styling - Force Dark Background & Pure White Text */
div[data-testid="file"], 
div[data-testid="file"] *, 
.file-upload, 
.file-upload *, 
.file-preview, 
.file-preview *, 
[data-testid="file-upload"], 
[data-testid="file-upload"] *, 
.upload-container, 
.upload-container *, 
table.files, 
table.files *, 
.file-item, 
.file-item * {
    background-color: #1f2937 !important;
    background: #1f2937 !important;
    color: #ffffff !important;
}

div[data-testid="file"], .file-upload, [data-testid="file-upload"], .upload-container {
    border: 2px dashed #6366f1 !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

div[data-testid="file"] .file-name, .file-preview .file-name, .file-name, .filename, td.filename, span.file-name, span.name {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

div[data-testid="file"] .file-size, .file-preview .file-size, .file-size, td.filesize {
    color: #a5b4fc !important;
    font-weight: 500 !important;
}

div[data-testid="file"] svg, .file-upload svg, .file-preview svg {
    stroke: #818cf8 !important;
}

div[data-testid="file"] button, .file-upload button, .file-preview button {
    background-color: #374151 !important;
    background: #374151 !important;
    color: #ffffff !important;
    border: 1px solid #4b5563 !important;
    border-radius: 6px !important;
}

input::placeholder, textarea::placeholder,
.gradio-textbox input::placeholder, .gradio-textbox textarea::placeholder,
div[data-testid="textbox"] textarea::placeholder {
    color: #9ca3af !important;
}

/* Accordion */
.chunks-accordion {
    border: 1.5px solid #4b5563 !important;
    background: #111827 !important;
    border-radius: 14px !important;
}

.chunks-accordion * {
    color: #ffffff !important;
}

/* Markdown Code Blocks */
pre, code, .markdown-body pre {
    background-color: #0f172a !important;
    color: #38bdf8 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* 1. 從根部篡改 Gradio 預設淺色變數，設為透明避免干擾 */
:root, .gradio-container {
    --color-accent-soft: transparent !important; 
}

/* 2. 聊天室主畫面大背景 */
div[data-testid="chatbot"], .chatbot, .bubble-wrap {
    background-color: #111827 !important;
    border: 1.5px solid #374151 !important;
    border-radius: 14px !important;
}

/* 3. 【破案關鍵】剝除外層行容器 (Row) 的武裝，讓它完全透明，不佔多餘空間 */
div[data-testid="chatbot"] .message-row {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* 4. 【精準打擊】只針對真正的對話氣泡 (.message) 上色與設定內距 */
div[data-testid="chatbot"] .message {
    background: #1f2937 !important;
    background-color: #1f2937 !important;
    border: 1.5px solid #374151 !important;
    border-radius: 12px !important;
    padding: 12px 18px !important; /* 設定正常的內距，解決上下留白過大 */
    box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
}

/* 5. 強制清空氣泡內部所有元素的殘留背景 */
div[data-testid="chatbot"] .message * {
    background: transparent !important;
    background-color: transparent !important;
}

/* 6. 設定文字為純白，但放過程式碼區塊 */
div[data-testid="chatbot"] .message *:not(code):not(pre) {
    color: #ffffff !important;
    line-height: 1.6 !important;
}

/* 7. 還原程式碼區塊該有的深色底與亮色字 */
div[data-testid="chatbot"] .message pre,
div[data-testid="chatbot"] .message code {
    background-color: #0f172a !important;
    color: #38bdf8 !important;
    border: 1px solid #334155 !important;
}

/* 8. 消除 Markdown 渲染產生的多餘空段落與外邊距 */
div[data-testid="chatbot"] .prose > *:first-child { margin-top: 0 !important; }
div[data-testid="chatbot"] .prose > *:last-child { margin-bottom: 0 !important; }
div[data-testid="chatbot"] .prose p:empty { display: none !important; }

"""

with gr.Blocks(title="AI 文件對話系統 (RAG Engine)") as demo:
    
    # Hero Header Banner (No Box Border & No Engine Badge)
    with gr.Row(elem_classes=["hero-banner"]):
        gr.HTML("""
            <div>
                <h1 class="hero-title">📄 AI 文件對話系統 (RAG Workstation)</h1>
                <p class="hero-subtitle">基於 Docling 結構化解析、BGE-M3 高維向量庫與 LangGraph 流程編排</p>
            </div>
        """)
    
    current_doc_id = gr.State("")
    
    with gr.Row():
        # Left Panel: Document Control Center (Adjusted smaller: scale=1)
        with gr.Column(scale=1, min_width=270, elem_classes=["glass-panel"]):
            gr.Markdown("### 📂 文件管理與索引狀態")
            file_input = gr.File(label="上傳自訂 PDF 檔案", file_types=[".pdf"])
            
            with gr.Row():
                upload_btn = gr.Button("📤 上傳並解析 PDF", variant="primary", elem_classes=["btn-primary"])
                default_btn = gr.Button("📑 載入預設 LightRAG 論文", elem_classes=["btn-preset"])
            reparse_btn = gr.Button("🔄 強制重新解析當前文件", elem_classes=["btn-preset"])
            
            gr.Markdown("---")
            gr.Markdown("#### 📊 向量庫狀態面板")
            status_output = gr.Textbox(label="系統處理狀態", interactive=False, placeholder="等待操作...")
            doc_id_display = gr.Textbox(label="Document ID (唯一識別碼)", interactive=False, placeholder="未指定文件")
            chunk_info = gr.Textbox(label="Chunk 切塊與快取資訊", interactive=False, placeholder="尚無資料")
            
        # Right Panel: Chat Workstation (Expanded larger: scale=3)
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            gr.Markdown("#### ⚡ 規格書預設測試集 (快捷提問)")
            with gr.Row():
                q1_btn = gr.Button("📝 1. Summary this document", elem_classes=["btn-preset"])
                q2_btn = gr.Button("⚖️ 2. Compare LightRAG with GraphRAG", elem_classes=["btn-preset"])
                q3_btn = gr.Button("📊 3. Performance of ablated versions", elem_classes=["btn-preset"])
                
            gr.Markdown("### 💬 對話與問答視窗")
            chatbot = gr.Chatbot(height=520, label="對話紀錄")
            
            with gr.Row():
                msg_input = gr.Textbox(placeholder="請輸入關於文件的問題 (例如: What is LightRAG?)...", label="提問內容")
                send_btn = gr.Button("💬 送出提問", variant="primary", elem_classes=["btn-primary"])
            
            with gr.Accordion("🔍 檢索段落與脈絡檢視器 (Retrieved Chunks Viewer)", open=True, elem_classes=["chunks-accordion"]):
                retrieved_chunks_view = gr.Markdown("*(發送提問後，LangGraph 流程檢索到的文件 Chunk 內文與頁碼將即時顯示於此)*")

    def sanitize_content(content):
        if not content:
            return ""
        if isinstance(content, list):
            return "".join(item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content).strip()
        
        c_str = str(content).strip()
        if c_str.startswith("[{'") and "'text':" in c_str:
            import ast
            try:
                parsed = ast.literal_eval(c_str)
                if isinstance(parsed, list):
                    return "".join(item.get("text", "") for item in parsed if isinstance(item, dict)).strip()
            except Exception:
                pass
        return c_str

    def clean_history(history):
        if not history:
            return []
        cleaned = []
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                cleaned.append({"role": str(item["role"]), "content": sanitize_content(item["content"])})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                if item[0]:
                    cleaned.append({"role": "user", "content": sanitize_content(item[0])})
                if item[1]:
                    cleaned.append({"role": "assistant", "content": sanitize_content(item[1])})
        return cleaned

    # 第一步：按下送出時，立即清空輸入框並在聊天視窗顯示使用者問題氣泡 (0.01秒)
    def user_submit(user_msg, history):
        history = clean_history(history)
        if not user_msg:
            return "", history
        history.append({"role": "user", "content": user_msg})
        return "", history

    # 第二步：背景發起 API 請求，生成回答後追加 Assistant 氣泡與 Chunk 檢視
    def bot_respond(history, doc_id):
        if not history or history[-1].get("role") != "user":
            return history, "*(無檢索內容)*"
        user_msg = history[-1].get("content", "")
        ans, chunks_display = chat_fn(user_msg, history[:-1], doc_id)
        history.append({"role": "assistant", "content": ans})
        return history, chunks_display

    # 快捷提問點擊時的即時 User 氣泡顯示
    def quick_q_submit(q_text, history):
        history = clean_history(history)
        history.append({"role": "user", "content": q_text})
        return history

    upload_btn.click(
        upload_file_to_backend, 
        inputs=[file_input], 
        outputs=[status_output, current_doc_id, chunk_info]
    ).then(
        lambda doc_id: doc_id, 
        inputs=[current_doc_id], 
        outputs=[doc_id_display]
    )

    default_btn.click(
        load_default_paper, 
        outputs=[status_output, current_doc_id, chunk_info]
    ).then(
        lambda doc_id: doc_id, 
        inputs=[current_doc_id], 
        outputs=[doc_id_display]
    )

    def reparse_current_doc(file_input_val):
        if file_input_val:
            return upload_file_to_backend(file_input_val, force_reparse=True)
        else:
            return load_default_paper(force_reparse=True)

    reparse_btn.click(
        reparse_current_doc, 
        inputs=[file_input], 
        outputs=[status_output, current_doc_id, chunk_info]
    ).then(
        lambda doc_id: doc_id, 
        inputs=[current_doc_id], 
        outputs=[doc_id_display]
    )

    # 綁定送出按鈕與 Enter 事件：先即時渲染 User 氣泡 ➡️ 再發起 RAG/LLM 回答
    send_btn.click(
        user_submit, 
        inputs=[msg_input, chatbot], 
        outputs=[msg_input, chatbot]
    ).then(
        bot_respond, 
        inputs=[chatbot, current_doc_id], 
        outputs=[chatbot, retrieved_chunks_view]
    )

    msg_input.submit(
        user_submit, 
        inputs=[msg_input, chatbot], 
        outputs=[msg_input, chatbot]
    ).then(
        bot_respond, 
        inputs=[chatbot, current_doc_id], 
        outputs=[chatbot, retrieved_chunks_view]
    )

    # 綁定快捷問題按鈕：先即時渲染 User 氣泡 ➡️ 再發起 RAG/LLM 回答
    q1_btn.click(
        lambda h: quick_q_submit("Summary this document", h), 
        inputs=[chatbot], 
        outputs=[chatbot]
    ).then(
        bot_respond, 
        inputs=[chatbot, current_doc_id], 
        outputs=[chatbot, retrieved_chunks_view]
    )

    q2_btn.click(
        lambda h: quick_q_submit("Compare LightRAG with GraphRAG", h), 
        inputs=[chatbot], 
        outputs=[chatbot]
    ).then(
        bot_respond, 
        inputs=[chatbot, current_doc_id], 
        outputs=[chatbot, retrieved_chunks_view]
    )

    q3_btn.click(
        lambda h: quick_q_submit("Performance of ablated versions of LightRAG", h), 
        inputs=[chatbot], 
        outputs=[chatbot]
    ).then(
        bot_respond, 
        inputs=[chatbot, current_doc_id], 
        outputs=[chatbot, retrieved_chunks_view]
    )

    # 頁面初始化時，自動預載預設論文 (利用 ChromaDB 快取秒載)
    demo.load(
        load_default_paper, 
        outputs=[status_output, current_doc_id, chunk_info]
    ).then(
        lambda doc_id: doc_id, 
        inputs=[current_doc_id], 
        outputs=[doc_id_display]
    )

if __name__ == "__main__":
    # 將 css 放在這裡，這是 Gradio 6.0 的新規定
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css)