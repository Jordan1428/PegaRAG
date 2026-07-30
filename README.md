# 🚀 PegaRAG：全端檢索增強生成系統 (End-to-End RAG System)

PegaRAG 是一個模組化、可擴展的檢索增強生成 (Retrieval-Augmented Generation) 解決方案。本專案將**核心 RAG 處理管線**與**全端使用者介面**完美解耦，支援強大的文件解析、圖結構 (Graph-based) 的推理流程，以及完整的系統評估 (Evaluation) 機制。

## ✨ 核心特色 (Features)

- **🧠 圖結構 RAG 處理管線 (Graph-based Pipeline)**：以 `graph.py`、`nodes.py` 與 `state.py` 為核心，採用狀態機與節點流程式設計，精準控制 LLM 的檢索與生成邏輯。
- **📄 智慧文件載入與解析 (Ingestion)**：內建專屬的 `parser.py` 與 `indexer.py`，並在後端搭配獨立的 `chunker.py`，支援高精度文件切塊與向量化。
- **📊 內建評估機制 (Evaluation)**：整合完整的 RAG 評估腳本 (`03_run_eval.py`)，可基於內建的 `eval_dataset.json/csv` 自動跑分並產出量化報告。
- **🖥️ 友善的 Web UI**：前後端分離架構。後端使用 FastAPI 獨立處理 `llm_client.py` 與 `vector_store.py`；前端則透過 `app.py` 提供基於 Gradio 打造的互動式對話介面。
- **⚙️ 自動化腳本支援**：提供完整的 scripts，一鍵完成索引建置 (`01_build_index.py`)、問答查詢 (`02_run_query.py`) 與資料庫狀態檢視 (`inspect_chroma.py`)。

---

## 📂 專案架構 (Project Structure)

專案主要分為 `RAG/` (核心演算法與管線) 與 `UI/` (Web 服務) 兩大區塊：

```text
PegaRAG/
├── RAG/                        # RAG 核心處理與評估模組
│   ├── eval_data/              # 評估用測試數據集 (csv/json)
│   ├── scripts/                # 自動化執行腳本 (建置索引、查詢、評估、檢視資料庫)
│   ├── src/                    # 核心原始碼
│   │   ├── ingestion/          # 資料載入與解析模組 (parser, indexer)
│   │   ├── pipeline/           # RAG 推理管線 (graph, nodes, state)
│   │   ├── prompts/            # LLM 提示詞管理 (qa_prompt.py)
│   │   ├── config.py           # 核心參數配置
│   │   └── llm_factory.py      # LLM 實例化工廠
│   ├── tests/                  # 單元測試模組 (test_ingestion.py, test_pipeline.py)
│   ├── .env.example            # 環境變數範例檔
│   └── requirements.txt        # RAG 模組專屬依賴套件
├── UI/                         # 系統操作介面與後端 API
│   ├── backend/                # FastAPI 後端 (main.py, llm_client.py, vector_store.py 等)
│   ├── frontend/               # Gradio 前端 UI (app.py)
│   ├── spec/                   # 系統規格書 (AI文件小試系統規格書.md)
│   ├── run.py                  # Web 服務一鍵啟動腳本
│   └── requirements.txt        # UI 專屬依賴套件
├── .gitignore                  # Git 忽略清單
└── README.md                   # 專案說明文件 (本文件)
```
*(註：系統執行時會自動生成 `chromadb/`、`uploads/` 以及 `eval_reports/` 等暫存或產出目錄，為保持版控整潔，這些目錄皆已加入 `.gitignore`。)*

---

## 🛠️ 系統安裝 (Installation)

### 1. 建立虛擬環境
建議使用 Python 3.8 以上版本的虛擬環境來隔離專案套件：
```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
venv\Scripts\activate

# 啟動虛擬環境 (macOS / Linux)
source venv/bin/activate
```

### 2. 安裝依賴套件
本專案的依賴套件分為核心模組與 Web 服務兩部分，請依序安裝：
```bash
# 安裝 RAG 核心模組套件
pip install -r RAG/requirements.txt

# 安裝 UI 與後端服務套件
pip install -r UI/requirements.txt
```

### 3. 環境變數設定
請複製範例設定檔，並填入你所需要的 API 金鑰及模型設定：
```bash
# 複製並建立 .env 檔案
cp RAG/.env.example RAG/.env
```
*(註：請在 `.env` 檔案中填寫 LLM 所需的 API Key 或是本地端模型的配置資訊。)*

---

## 🚀 快速啟動 (Quick Start)

本系統提供兩種操作模式，你可以根據需求選擇：

### 模式一：命令列操作 (CLI Mode)

若你想直接在終端機測試 RAG 核心能力，可以使用 `RAG/scripts/` 提供的自動化腳本：

1. **建立知識庫索引** (將文件解析、切塊、向量化並存入資料庫)：
   ```bash
   python RAG/scripts/01_build_index.py
   ```
2. **執行終端機查詢** (對知識庫進行問答)：
   ```bash
   python RAG/scripts/02_run_query.py
   ```
3. **執行系統評估** (跑分並產出報告)：
   ```bash
   python RAG/scripts/03_run_eval.py
   ```
*(你可以透過 `python RAG/scripts/inspect_chroma.py` 來檢視目前向量資料庫的儲存狀態)*

### 模式二：網頁介面操作 (Web UI Mode)

若你想透過視覺化介面操作、上傳文件並進行對話，請透過一鍵腳本啟動 Web 服務：

```bash
python UI/run.py
```
執行此腳本後，系統將自動依序啟動：
1. **FastAPI 核心後端**：預設運行於 `http://127.0.0.1:8000` (可瀏覽 `/docs` 查看 Swagger API 文件)。
2. **Gradio 聊天介面**：預設運行於 `http://127.0.0.1:7860` (系統會自動在預設瀏覽器中為你開啟介面)。
