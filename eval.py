"""Evaluate retrieval and answer quality against questions.json.

Metrics:
  - Retrieval hit rate: the correct (file, page) appears in the top-k chunks.
  - Answer accuracy: Claude acts as judge, comparing the answer to your expected answer.

Usage:
    python eval.py --db chroma_500
    python eval.py --db chroma_250
    python eval.py --db chroma_500 --no-judge     # retrieval only, no extra API cost
"""
import argparse
import json

from ask import RAG, CLAUDE_MODEL

JUDGE_PROMPT = """You are grading a question-answering system.

Question: {q}
Expected answer: {expected}
System answer: {got}

Does the system answer convey the same key facts as the expected answer?
Reply with exactly one word: CORRECT or INCORRECT."""


def judge(rag, q, expected, got):
    msg = rag.llm.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=5,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(q=q, expected=expected, got=got)}],
    )
    return msg.content[0].text.strip().upper().startswith("CORRECT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chroma_500")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()

    with open(a.questions, encoding="utf-8") as f:
        qs = json.load(f)
    qs = [q for q in qs if not q["question"].startswith("EXAMPLE")]
    if not qs:
        raise SystemExit("Add your real questions to questions.json first.")

    rag = RAG(a.db, a.k)
    hits = correct = 0
    rows = []
    for i, item in enumerate(qs, 1):
        answer, chunks = rag.answer(item["question"])
        hit = any(c["file"] == item["file"] and c["page"] == item["page"] for c in chunks)
        ok = None
        if not a.no_judge:
            ok = judge(rag, item["question"], item["expected_answer"], answer)
            correct += ok
        hits += hit
        rows.append({"question": item["question"], "hit": hit, "correct": ok, "answer": answer})
        print(f"{i:>2}. hit={'Y' if hit else 'N'} correct={ok if ok is not None else '-'}  {item['question'][:60]}")

    n = len(qs)
    print("\n" + "=" * 50)
    print(f"DB: {a.db}   k={a.k}   questions={n}")
    print(f"Retrieval hit rate: {hits / n:.0%}  ({hits}/{n})")
    if not a.no_judge:
        print(f"Answer accuracy:    {correct / n:.0%}  ({correct}/{n})")

    out = f"results_{a.db}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"db": a.db, "k": a.k, "n": n, "hit_rate": hits / n,
                   "accuracy": None if a.no_judge else correct / n, "rows": rows}, f, indent=2)
    print(f"Saved details to {out}")


if __name__ == "__main__":
    main()
