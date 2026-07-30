# 「讓AI跟文件對話」系統規格書

## 1. 專案背景與限制

本專案目標為打造一個網站聊天介面，使用者上傳文件後可針對文件內容提問，並由本地LLM根據檢索到的文件內容回答問題。

**核心限制：**
1. 不可使用現成的GPT/Gemini等雲端API解析文件，LLM僅能做text-to-text生成，不可依賴多模態雲端模型讀取PDF排版、圖表。
2. 假定LLM最大context size為10,000 tokens，所有送入LLM的prompt（含檢索內容、問題、系統指令）需控制在此範圍內。

**預設測試文件：** LightRAG論文 (arXiv:2410.05779)

**預設測試問題：**
1. Summary this document.
2. Compare LightRAG with GraphRAG.
3. Performance of ablated versions of LightRAG.

## 2. 技術選型與比較

| 層級 | 選用框架 | 主要替代方案 | 選用理由 |
|---|---|---|---|
| 前端 UI | Gradio | Streamlit | 內建ChatInterface與檔案上傳元件，開發速度快，適合demo型專案，不需自行處理頁面刷新狀態管理 |
| 後端 API | FastAPI | Flask | 原生支援async，適合處理LLM生成的非同步/串流回應；自動生成API文件，方便驗證contract |
| 向量儲存 | Chroma | Qdrant, FAISS | 零基礎設施、Python內直接import即可用，適合單篇文件、小規模資料的prototype；Qdrant的高併發與大規模擴展優勢在此專案用不到 |


## 3. 系統架構

```
使用者瀏覽器
    │
    ▼
[Gradio 前端] ──HTTP──▶ [FastAPI 後端]
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        [PDF解析模組]   [切塊+Embedding]   [Chroma向量庫]
              │               │               │
              └───────┬───────┴───────┬───────┘
                      ▼               ▼
                [檢索模組 Top-K]  [Map-Reduce摘要模組]
                      │               │
                      └───────┬───────┘
                              ▼
                        [本地LLM (Ollama)]
                              │
                              ▼
                        回傳答案給前端
```

## 4. API Contract

系統僅需兩個核心Endpoint，前後端依此規格各自平行開發。

### 4.1 POST /upload

**用途：** 上傳PDF文件，後端進行解析、切塊、embedding並存入向量庫。

**Request:**
```json
{
  "file": "<PDF binary, multipart/form-data>"
}
```

**Response (成功):**
```json
{
  "document_id": "doc_001",
  "status": "success",
  "chunk_count": 128
}
```

**Response (失敗):**
```json
{
  "status": "error",
  "message": "無法解析此PDF文件"
}
```

### 4.2 POST /chat

**用途：** 針對已上傳文件提問，後端檢索相關內容並組成prompt送入LLM生成回答。

**Request:**
```json
{
  "document_id": "doc_001",
  "question": "summary this document"
}
```

**Response:**
```json
{
  "answer": "本篇論文提出LightRAG，是一種結合圖結構與向量檢索的RAG框架...",
  "source_chunks": ["chunk_012", "chunk_045"]
}
```

**欄位型別說明：**

| 欄位 | 型別 | 說明 |
|---|---|---|
| document_id | string | 文件唯一識別碼，上傳後由後端產生 |
| status | string | success 或 error |
| chunk_count | integer | 切塊後總片數，供debug參考 |
| question | string | 使用者輸入的問題文字 |
| answer | string | LLM生成的最終回答 |
| source_chunks | array[string] | 回答所依據的檢索片段ID，供追溯來源 |

前端在後端邏輯完成前，可先以上述格式產生假資料(mock response)，先行完成上傳畫面與聊天框互動，待後端API實作完成後再替換為真實請求。

## 5. 核心技術處理邏輯

### 5.1 文件切塊策略（應對10K context限制）

- 使用PyMuPDF/pdfplumber解析PDF為結構化文字，盡量保留章節標題與表格結構(轉為markdown表格)。
- 依語義或段落邊界將全文切成固定大小chunk（例如每塊300-500 tokens），並保留少量overlap以避免語義斷裂。
- 每個chunk連同其embedding向量與metadata（章節、頁碼）存入Chroma。
- 提問時，先將問題embedding，取回Top-K（例如3-5個）最相關chunk，組成的prompt總長度需控制在10K token以內，扣除系統指令與問題本身留出的空間。

### 5.2 Summary處理邏輯（Map-Reduce摘要）

- 若使用者要求整份文件摘要，單次檢索無法涵蓋全文，改採Map-Reduce策略：
  1. **Map階��：** 將全文所有chunk逐一送入LLM，各自產生短摘要。
  2. **Reduce階段：** 將所有chunk摘要合併，若合併後仍超過10K token，遞迴重複摘要壓縮，直到總長度可一次送入LLM生成最終總結。

### 5.3 針對三個測試問題的因應設計

| 問題 | 挑戰 | 因應方式 |
|---|---|---|
| Summary this document | 全文遠超context上限 | 採用5.2 Map-Reduce摘要流程 |
| Compare LightRAG with GraphRAG | 跨章節資訊整合 | 對關鍵詞多次檢索，取回相關段落後交由LLM綜合比較 |
| Performance of ablated versions | 資訊集中於實驗/表格章節 | PDF解析時保留表格結構，並針對"ablation"、"performance"等關鍵詞加強檢索權重 |

## 6. 交付項目

1. 完整程式碼（zip或git連結）。
2. PPT：說明專案架構與結果、與AI協作過程、協作心得與學習。
3. Demo：本地運行即可，無需部署上雲。
