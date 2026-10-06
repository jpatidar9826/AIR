"""Policy document ingestion, chunking and retrieval (TF-IDF, optional embeddings)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}


@dataclass
class Chunk:
    doc: str
    section: str
    text: str
    chunk_id: int


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        import docx
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return ""


def chunk_document(doc_name: str, text: str, size=config.RAG_CHUNK_SIZE,
                   overlap=config.RAG_CHUNK_OVERLAP) -> list[Chunk]:
    """Split on headings/paragraphs, then pack paragraphs into ~size-char chunks."""
    chunks, section, buf = [], doc_name, ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(Chunk(doc_name, section, buf.strip(), len(chunks)))
        buf = buf[-overlap:] if overlap else ""

    for para in paragraphs:
        if para.startswith("#"):
            flush(); buf = ""
            section = para.lstrip("# ").split("\n")[0]
            para = "\n".join(para.split("\n")[1:]).strip()
            if not para:
                continue
        while len(para) > size:  # very long paragraph
            buf += " " + para[:size]; para = para[size - overlap:]; flush()
        if len(buf) + len(para) > size:
            flush()
        buf += ("\n" if buf else "") + para
    flush()
    return chunks


class PolicyStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = None
        self.embeddings = None
        self._encoder = None

    def build(self, policy_dir: Path = config.POLICY_DIR) -> "PolicyStore":
        files = sorted(p for p in Path(policy_dir).rglob("*")
                       if p.suffix.lower() in SUPPORTED and not p.name.lower().startswith("readme"))
        if not files:
            raise FileNotFoundError(f"No policy documents found in {policy_dir}")
        for f in files:
            self.chunks.extend(chunk_document(f.name, read_document(f)))
        texts = [f"{c.section}. {c.text}" for c in self.chunks]
        self.matrix = self.vectorizer.fit_transform(texts)
        if config.USE_EMBEDDINGS:
            try:
                self.embeddings = self._encode(texts)
            except Exception as exc:
                print(f"Embeddings disabled ({exc}); using TF-IDF only")
        return self

    def _encode(self, texts):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._encoder.encode(texts, normalize_embeddings=True)

    def search(self, query: str, k: int = config.RAG_TOP_K) -> list[tuple[Chunk, float]]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        if self.embeddings is not None:
            emb = np.asarray(self.embeddings) @ self._encode([query])[0]
            scores = 0.5 * scores + 0.5 * emb
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path=config.RAG_INDEX_PATH):
        enc, self._encoder = self._encoder, None
        joblib.dump(self, path)
        self._encoder = enc

    @staticmethod
    def load_or_build(path=config.RAG_INDEX_PATH) -> "PolicyStore":
        if Path(path).exists():
            return joblib.load(path)
        store = PolicyStore().build()
        store.save(path)
        return store

    @property
    def documents(self) -> list[str]:
        return sorted({c.doc for c in self.chunks})
