"""Retrieve relevant chunks and have Claude answer using only that context.

Usage:
    python ask.py "What is the attendance requirement?"
    python ask.py --db chroma_250 "your question"
"""
import argparse
import os

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "docs"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM = (
    "You answer questions using ONLY the provided context excerpts. "
    "Cite every claim as [file, page]. If the answer is not in the context, "
    "say you don't know. Do not use outside knowledge."
)


class RAG:
    def __init__(self, db="chroma_500", k=5):
        self.k = k
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.col = chromadb.PersistentClient(path=db).get_collection(COLLECTION)
        self.llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def retrieve(self, question):
        emb = self.embedder.encode([question]).tolist()
        res = self.col.query(query_embeddings=emb, n_results=self.k)
        return [
            {"text": t, "file": m["file"], "page": m["page"]}
            for t, m in zip(res["documents"][0], res["metadatas"][0])
        ]

    def answer(self, question):
        chunks = self.retrieve(question)
        context = "\n\n".join(
            f"[{c['file']}, page {c['page']}]\n{c['text']}" for c in chunks
        )
        msg = self.llm.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            system=SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }],
        )
        return msg.content[0].text, chunks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--db", default="chroma_500")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    text, chunks = RAG(a.db, a.k).answer(a.question)
    print(text)
    print("\nSources:")
    for c in chunks:
        print(f"  - {c['file']}, page {c['page']}")
