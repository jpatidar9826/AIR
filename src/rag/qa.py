"""Retrieval-augmented question answering over HR policies."""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from src.llm.client import LLMClient
from src.llm.prompts import RAG_SYSTEM, RAG_USER
from src.rag.store import PolicyStore

NOT_FOUND = "I couldn't find this in the HR policies. Please contact HR."


def _extractive_answer(question: str, hits, max_sentences: int = 3) -> str:
    """Offline fallback: pick the sentences most similar to the question."""
    sentences = []
    for idx, (chunk, _) in enumerate(hits, start=1):
        for s in re.split(r"(?<=[.!?])\s+|\n+", chunk.text):
            s = s.strip(" -*")
            if len(s.split()) >= 5:
                sentences.append((s, idx))
    if not sentences:
        return NOT_FOUND
    vec = TfidfVectorizer(stop_words="english").fit([s for s, _ in sentences] + [question])
    sims = (vec.transform([s for s, _ in sentences]) @ vec.transform([question]).T).toarray().ravel()
    best = sorted(np.argsort(sims)[::-1][:max_sentences])
    return " ".join(f"{sentences[i][0]} [{sentences[i][1]}]" for i in best if sims[i] > 0) or NOT_FOUND


def answer_question(question: str, store: PolicyStore, llm: LLMClient | None = None,
                    k: int = config.RAG_TOP_K) -> dict:
    hits = [(c, s) for c, s in store.search(question, k) if s >= config.RAG_MIN_SCORE]
    sources = [{"n": i, "doc": c.doc, "section": c.section, "score": round(s, 3), "text": c.text}
               for i, (c, s) in enumerate(hits, start=1)]
    if not hits:
        return {"answer": NOT_FOUND, "sources": [], "mode": "none"}

    if llm is not None and llm.available:
        context = "\n\n".join(f"[{s['n']}] ({s['doc']} - {s['section']})\n{s['text']}" for s in sources)
        reply = llm.chat(RAG_SYSTEM, RAG_USER.format(context=context, question=question), purpose="rag_answer")
        if reply:
            return {"answer": reply, "sources": sources, "mode": "llm"}
    return {"answer": _extractive_answer(question, hits), "sources": sources, "mode": "extractive"}
