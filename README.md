# Document Q&A: RAG Chatbot with Evaluation

Ask questions about a set of PDFs. Answers come only from the documents and cite `[file, page]`.
Includes an evaluation harness that measures retrieval hit rate and answer accuracy.

**Live demo:** _add your link_

## Architecture

PDFs -> split into overlapping word chunks -> embed (`all-MiniLM-L6-v2`) -> Chroma vector store.
Question -> embed -> top-5 chunks -> Claude answers using only those chunks, with citations.

## Results

| Config | Retrieval hit rate | Answer accuracy |
|---|---|---|
| Chunk 500 / overlap 50 | _ % | _ % |
| Chunk 250 / overlap 25 | _ % | _ % |

_Evaluated on N hand-written questions. Fill in after running `eval.py`._

## Run it

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key  # Windows PowerShell: $env:ANTHROPIC_API_KEY="your_key"

# 1. put text-based PDFs in data/
python ingest.py                           # chunk 500 -> chroma_500/
python ingest.py --chunk 250 --overlap 25  # chunk 250 -> chroma_250/

# 2. try it
python ask.py "your question"

# 3. evaluate (edit questions.json first)
python eval.py --db chroma_500
python eval.py --db chroma_250

# 4. web UI
streamlit run app.py
```

## Deploy on Hugging Face Spaces

1. Create a new Space, choose the **Streamlit** SDK.
2. Push this repo to it, including your `data/` PDFs. Add this block at the very top of `README.md`:
   ```
   ---
   title: Document QA
   sdk: streamlit
   app_file: app.py
   ---
   ```
3. In Space Settings -> Secrets, add `ANTHROPIC_API_KEY`.
4. The index is built automatically on first launch.

## What I learned

_Write 3-4 sentences: what chunk size worked best, where retrieval failed, what you would improve
(hybrid search, reranking, better PDF parsing)._
