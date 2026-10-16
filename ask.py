"""Retrieve relevant chunks and have an LLM answer using only that context.

Usage:
    python ask.py "What is the attendance requirement?"
    python ask.py --db chroma_250 "your question"
"""
import argparse
import os

import time

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "docs"
# Free by default: Groq free tier. Set LLM_PROVIDER=anthropic to use Claude instead.
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
MODELS = {"groq": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
          "anthropic": os.getenv("LLM_MODEL", "claude-sonnet-5-5")}
LLM_MODEL = MODELS[PROVIDER]

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
        self.llm = self._make_client()

    @staticmethod
    def _make_client():
        if PROVIDER == "groq":
            from groq import Groq
            return Groq()  # reads GROQ_API_KEY
        import anthropic
        return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(self, system, user, max_tokens=700):
        """One LLM call, provider-agnostic, with retry on rate limits."""
        for attempt in range(4):
            try:
                if PROVIDER == "groq":
                    msgs = ([{"role": "system", "content": system}] if system else [])
                    msgs.append({"role": "user", "content": user})
                    r = self.llm.chat.completions.create(
                        model=LLM_MODEL, messages=msgs, max_tokens=max_tokens, temperature=0)
                    return r.choices[0].message.content
                r = self.llm.messages.create(
                    model=LLM_MODEL, max_tokens=max_tokens, system=system or "You are a grader.",
                    messages=[{"role": "user", "content": user}])
                return r.content[0].text
            except Exception as e:
                msg = str(e).lower()
                if attempt == 3 or "401" in msg or "auth" in msg or "api key" in msg:
                    raise
                time.sleep(8 * (attempt + 1))  # free tiers rate-limit: back off and retry

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
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        text = self.complete(SYSTEM, prompt, 700)
        return text, chunks


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
