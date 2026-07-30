# RAG 核心處理模組 (RAG Core Processing Module)

本專案採用 **「LlamaIndex 管理資料 + LangGraph 控制流程」** 的分層架構，專為高精度文件解析（包含表格與結構化特徵）、向量檢索與 LLM 流程編排而設計。

---

## 🌟 系統特色與技術選型

1. **文件解析與分塊 (Parser & Chunker)**:
   - **Docling + HybridChunker**: 完整提取標題、段落與 Markdown/HTML 表格結構，並採用 tokenizer-aware 切割 (`repeat_table_header=True`)。
2. **向量模型 (Embedding Model)**:
   - **BGE-M3 (`BAAI/bge-m3`)**: 本機 HuggingFace 執行，支援 8192 長文本 context，無 API 費用。
3. **向量資料庫 (Vector Store)**:
   - **ChromaDB**: 本機持久化儲存 (`data/chromadb`)，無需額外啟動服務。
4. **流程編排 (Orchestration)**:
   - **LangGraph**: 採用 `StateGraph` 編排 RAG 流程 (`START -> retrieve_node -> generate_node -> END`)。
5. **LLM 彈性介面 (Factory Pattern)**:
   - 支援通過 `.env` 或 `config.py` 動態切換：**Ollama (本機 llama3 等)**、**OpenAI (gpt-4o-mini)**、**Gemini**、**Anthropic (Claude)** 或 **Mock (離線測試)**。
6. **量化評測模組 (Evaluation Module)**:
   - 針對 10 題測試集自動計算 **Context Relevance (相關性)**、**Answer Faithfulness (忠實度)** 與 **Answer Correctness (正確性)**，產出 CSV / JSON 報告。

---

## 📁 專案檔案結構 (Project Structure)

```text
rag_assignment/
├── data/
│   ├── raw/                  # 原始 PDF 檔案存放區 (如 2410.05779v3.pdf)
│   └── chromadb/             # ChromaDB 本機持久化資料夾 (自動生成)
│
├── eval_data/
│   ├── eval_dataset.csv      # 10 題測試題目與 Ground Truth (CSV)
│   ├── eval_dataset.json     # 10 題測試題目與 Ground Truth (JSON)
│   └── eval_reports/         # 評估腳本產出的評分報告 (eval_report.csv / eval_report.json)
│
├── src/                      # 核心程式碼目錄
│   ├── __init__.py
│   ├── config.py             # 統管環境變數與系統參數 (Top-K, Chunk Size 等)
│   ├── llm_factory.py        # LLM 介面工廠 (Ollama / OpenAI / Gemini / Anthropic / Mock)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py         # Docling 解析與 HybridChunker 邏輯
│   │   └── indexer.py        # 負責將 Chunk 轉為 TextNode 並存入 ChromaDB 建立 Index
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── state.py          # 定義 LangGraph 的 TypedDict State schema
│   │   ├── nodes.py          # 實作 retrieve_node 與 generate_node 邏輯
│   │   └── graph.py          # 將 nodes 組合並編譯為 LangGraph 執行實例
│   └── prompts/
│       └── qa_prompt.py      # 統一存放 RAG 生成用的 Prompt Template
│
├── tests/                    # 單元測試 (Unit Tests)
│   ├── test_ingestion.py
│   └── test_pipeline.py
│
├── scripts/
│   ├── 01_build_index.py     # 執行腳本：讀取 PDF、解析分塊、建立本機向量庫
│   ├── 02_run_query.py       # 執行腳本：單次問答測試 (CLI 互動或參數輸入)
│   └── 03_run_eval.py        # 執行腳本：跑 10 題測試集並產出評估分數報告
│
├── .env.example              # 環境變數範例
├── .env                      # 實際運行環境設定檔
├── requirements.txt          # 依賴套件清單
└── README.md                 # 專案啟動與操作說明
```

---

## ⚙️ 環境安裝 (Environment Setup)

建議使用 Conda 建立 Python 3.12 獨立虛擬環境：

```bash
# 1. 建立 Conda 虛擬環境 (Python 3.12)
conda create -n rag_env python=3.12 -y

# 2. 啟動環境
conda activate rag_env

# 3. 安裝專案依賴
pip install -r requirements.txt
```

---

## 🚀 設定與運行步驟 (Quickstart & Usage)

### 1. 配置環境變數 (`.env`)
複製 `.env.example` 為 `.env` 並根據需求修改設定：

```bash
cp .env.example .env
```

`.env` 設定範例：
```ini
LLM_TYPE=openai                 # 可選: openai, ollama, gemini, anthropic, mock
LLM_MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL_NAME=BAAI/bge-m3
TOP_K=5
```

---

### 2. 建立向量庫索引 (Build Vector Index)
將 target PDF 放入 `data/raw/` 目錄中 (如 `2410.05779v3.pdf`)，執行：

```bash
python scripts/01_build_index.py
```
*腳本會透過 Docling 進行結構化解析與 HybridChunker 切割，並計算 BGE-M3 Embedding 寫入 `data/chromadb/`。*

---

### 3. 執行 RAG 問答測試 (Run RAG Query)

#### (A) 指定單一問題：
```bash
python scripts/02_run_query.py --query "Summarize this document."
```

#### (B) 進入 CLI 互動模式：
```bash
python scripts/02_run_query.py
```

---

### 4. 執行評測模組 (Run System Evaluation)
針對 `eval_data/eval_dataset.csv` 中的 10 題測試題與 Ground Truth 進行量化評估：

```bash
python scripts/03_run_eval.py
```
評估完成後會於 `eval_data/eval_reports/` 生成：
- `eval_report.csv`: 包含各題單項分數 (Context Relevance, Answer Faithfulness, Answer Correctness) 與整體平均分。
- `eval_report.json`: JSON 格式的詳細評測紀錄。

---

### 5. 運行單元測試 (Unit Tests)
使用 Pytest 驗證基礎模組與流程 correctness：

```bash
pytest tests/
```
