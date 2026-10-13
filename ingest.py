"""Read PDFs, split into overlapping word chunks, embed them, store in Chroma.

Usage:
    python ingest.py                         # chunk 500, overlap 50 -> chroma_500/
    python ingest.py --chunk 250 --overlap 25  # -> chroma_250/
"""
import argparse
import glob
import os

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "docs"


def chunk_words(text, size, overlap):
    """Split text into chunks of `size` words, each overlapping the last by `overlap`."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) < 20 and chunks:  # drop tiny trailing scraps
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def build(data_dir="data", db=None, chunk=500, overlap=50):
    db = db or f"chroma_{chunk}"
    pdfs = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs found in '{data_dir}/'. Add some and try again.")

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=db)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for path in pdfs:
        fname = os.path.basename(path)
        reader = PdfReader(path)
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            for i, piece in enumerate(chunk_words(text, chunk, overlap)):
                ids.append(f"{fname}-p{page_no}-c{i}")
                docs.append(piece)
                metas.append({"file": fname, "page": page_no})
        print(f"Read {fname}: {len(reader.pages)} pages")

    if not docs:
        raise SystemExit("No text extracted. Your PDFs may be scans; use text-based PDFs.")

    batch = 64
    for i in range(0, len(docs), batch):
        embs = model.encode(docs[i:i + batch]).tolist()
        col.add(ids=ids[i:i + batch], documents=docs[i:i + batch],
                metadatas=metas[i:i + batch], embeddings=embs)
    print(f"Stored {len(docs)} chunks in '{db}' (chunk={chunk}, overlap={overlap})")
    return db


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--chunk", type=int, default=500, help="words per chunk")
    ap.add_argument("--overlap", type=int, default=50, help="overlapping words")
    ap.add_argument("--db", default=None, help="Chroma folder (default chroma_<chunk>)")
    a = ap.parse_args()
    build(a.data, a.db, a.chunk, a.overlap)
