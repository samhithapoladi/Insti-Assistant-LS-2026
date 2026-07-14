import os
import re
import json
import glob
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from google import genai
except ImportError:
    genai = None


class Config:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    CHUNK_SIZE = 800        
    CHUNK_OVERLAP = 150     

    TOP_K = 4               
    MIN_SIMILARITY = 0.25   

    GEMINI_MODEL = "gemini-3.5-flash"   

    INDEX_DIR = "vector_store"
    FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
    METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")


@dataclass
class RawDocument:
    text: str
    source: str          
    page: Optional[int] = None


def _read_pdf(path: str) -> List[RawDocument]:
    if PdfReader is None:
        raise ImportError("pypdf is not installed. Run: pip install pypdf")
    reader = PdfReader(path)
    docs = []
    fname = os.path.basename(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            docs.append(RawDocument(text=text, source=fname, page=i + 1))
    return docs


def _read_text_file(path: str) -> List[RawDocument]:
    fname = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return [RawDocument(text=text, source=fname)] if text else []


def load_documents(doc_dir: str) -> List[RawDocument]:

    if not os.path.isdir(doc_dir):
        raise FileNotFoundError(f"Document folder not found: {doc_dir}")

    all_docs: List[RawDocument] = []
    paths = sorted(
        glob.glob(os.path.join(doc_dir, "*.pdf"))
        + glob.glob(os.path.join(doc_dir, "*.txt"))
        + glob.glob(os.path.join(doc_dir, "*.md"))
    )

    if not paths:
        raise ValueError(
            f"No .pdf/.txt/.md files found in '{doc_dir}'. "
            "Add at least 5 source documents before building the index."
        )

    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                all_docs.extend(_read_pdf(path))
            else:
                all_docs.extend(_read_text_file(path))
        except Exception as e:
            print(f"[WARN] Skipping '{path}': {e}")

    return all_docs


@dataclass
class Chunk:
    text: str
    source: str
    page: Optional[int]
    chunk_id: int


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_documents(
    raw_docs: List[RawDocument],
    chunk_size: int = Config.CHUNK_SIZE,
    overlap: int = Config.CHUNK_OVERLAP,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 0

    for doc in raw_docs:
        text = _clean_text(doc.text)
        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end]
            if piece.strip():
                chunks.append(
                    Chunk(
                        text=piece.strip(),
                        source=doc.source,
                        page=doc.page,
                        chunk_id=chunk_id,
                    )
                )
                chunk_id += 1
            if end >= len(text):
                break
            start = end - overlap  

    return chunks


class Embedder:
    """Thin wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str = Config.EMBEDDING_MODEL_NAME):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return embeddings / norms


class VectorStore:

    def __init__(self, dim: int):
        if faiss is None:
            raise ImportError("faiss is not installed. Run: pip install faiss-cpu")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim) 
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]):
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = Config.TOP_K):
        """Returns list of (Chunk, similarity_score), best first."""
        scores, idxs = self.index.search(
            query_embedding.astype("float32").reshape(1, -1), top_k
        )
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, index_dir: str = Config.INDEX_DIR):
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, index_dir: str = Config.INDEX_DIR):
        if faiss is None:
            raise ImportError("faiss is not installed. Run: pip install faiss-cpu")
        index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
            chunks = pickle.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.chunks = chunks
        return store


def build_index(doc_dir: str, index_dir: str = Config.INDEX_DIR) -> VectorStore:
    """End-to-end: load docs -> chunk -> embed -> build FAISS index -> save."""
    print(f"Loading documents from '{doc_dir}'...")
    raw_docs = load_documents(doc_dir)
    print(f"  Loaded {len(raw_docs)} raw pages/files.")

    print("Chunking...")
    chunks = chunk_documents(raw_docs)
    print(f"  Produced {len(chunks)} chunks.")
    if not chunks:
        raise ValueError("No chunks were produced — check your source documents.")

    print(f"Embedding with '{Config.EMBEDDING_MODEL_NAME}'...")
    embedder = Embedder()
    embeddings = embedder.embed([c.text for c in chunks])

    print("Building FAISS index...")
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)
    store.save(index_dir)
    print(f"Index saved to '{index_dir}/'.")
    return store


def load_or_build_index(doc_dir: str, index_dir: str = Config.INDEX_DIR, force_rebuild: bool = False) -> VectorStore:
    have_index = os.path.exists(os.path.join(index_dir, "index.faiss"))
    if have_index and not force_rebuild:
        print(f"Loading existing index from '{index_dir}/'...")
        return VectorStore.load(index_dir)
    return build_index(doc_dir, index_dir)


REFUSAL_MESSAGE = "I don't know based on the available documents."

SYSTEM_INSTRUCTIONS = f"""You are IITB Insti-Assist, a factual assistant that answers \
questions ONLY using the CONTEXT provided below, which was retrieved from official \
IIT Bombay documents.

Rules you must follow strictly:
1. Base your answer ONLY on the CONTEXT. Do not use outside knowledge, do not guess, \
and do not fill gaps with assumptions.
2. If the CONTEXT does not contain enough information to answer the question, \
respond with EXACTLY this sentence and nothing else: "{REFUSAL_MESSAGE}"
3. When you do answer, be concise and specific, and where useful quote or closely \
paraphrase the relevant policy/number/rule rather than being vague.
4. Do not mention "the context" or "the documents" explicitly in your answer — just \
answer naturally, as if you know this about IIT Bombay.
"""


class GeminiRAG:

    def __init__(
        self,
        vector_store: VectorStore,
        api_key: str,
        model: str = Config.GEMINI_MODEL,
        top_k: int = Config.TOP_K,
        min_similarity: float = Config.MIN_SIMILARITY,
    ):
        if genai is None:
            raise ImportError("google-genai is not installed. Run: pip install google-genai")
        if not api_key:
            raise ValueError("A Gemini API key is required.")

        self.store = vector_store
        self.embedder = Embedder()
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str):
        query_emb = self.embedder.embed([query])[0]
        results = self.store.search(query_emb, top_k=self.top_k)
        return [(chunk, score) for chunk, score in results if score >= self.min_similarity]

    def answer(self, query: str) -> Dict:
        """
        Returns:
          {
            "answer": str,
            "sources": [ {"source": str, "page": int|None, "text": str}, ... ],
            "grounded": bool   # False if we refused / no relevant context found
          }
        """
        retrieved = self.retrieve(query)

        if not retrieved:
            return {"answer": REFUSAL_MESSAGE, "sources": [], "grounded": False}

        context_block = "\n\n".join(
            f"[Source: {c.source}"
            + (f", page {c.page}" if c.page else "")
            + f"]\n{c.text}"
            for c, _ in retrieved
        )

        prompt = (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        answer_text = (response.text or "").strip()

        sources = [
            {
                "source": c.source,
                "page": c.page,
                "text": c.text,
            }
            for c, score in retrieved
        ]

        grounded = REFUSAL_MESSAGE not in answer_text

        return {"answer": answer_text, "sources": sources if grounded else [], "grounded": grounded}


if __name__ == "__main__":
    DOC_DIR = "docs"
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

    if not API_KEY:
        print(
            "Set GEMINI_API_KEY as an environment variable before running, e.g.\n"
            "  export GEMINI_API_KEY='your-key-here'   # mac/linux\n"
            "  setx GEMINI_API_KEY \"your-key-here\"      # windows\n"
        )
    else:
        store = load_or_build_index(DOC_DIR)
        rag = GeminiRAG(vector_store=store, api_key=API_KEY)

        print("\nIITB Insti-Assist (CLI mode). Type 'exit' to quit.\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            result = rag.answer(q)
            print(f"\nAssistant: {result['answer']}\n")
            if result["sources"]:
                print("Sources used:")
                for s in result["sources"]:
                    page_str = f", page {s['page']}" if s["page"] else ""
                    print(f"  - {s['source']}{page_str}")
                    print(f"      \"{s['text']}\"")
            print()