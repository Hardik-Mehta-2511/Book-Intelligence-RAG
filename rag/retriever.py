import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = Path("data/processed/books_embeddings.npz")
META_FILE = Path("data/processed/books_embeddings_meta.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
_SBERT = None


def load_embeddings():
    data = np.load(EMBEDDINGS_FILE)
    return data["embeddings"]


def load_meta():
    return pd.read_csv(META_FILE, encoding="utf-8")


def embed_query(query: str):
    global _SBERT
    if _SBERT is None:
        _SBERT = SentenceTransformer(MODEL_NAME)
    return _SBERT.encode([query], convert_to_numpy=True)[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)


def search(query: str, top_k: int = 5, filters: dict = None, min_score: float = None):
    """Search for nearest books to `query`.

    Args:
        query: text query
        top_k: number of results to return
        filters: optional dict with keys like `min_price`, `max_price`, `min_rating`, `genre`.
        min_score: optional minimum cosine similarity threshold (0-1)

    Returns a DataFrame with matched items and a `score` column.
    """
    embeddings = load_embeddings()
    meta = load_meta()

    query_vec = embed_query(query)
    scores = cosine_similarity(embeddings, query_vec)

    # Apply filters by masking scores where items do not match
    if filters:
        mask = np.ones(len(scores), dtype=bool)

        if "min_price" in filters and "price" in meta.columns:
            try:
                prices = pd.to_numeric(meta["price"].astype(str).str.replace("[^0-9.]+", ""), errors="coerce")
                mask &= prices >= float(filters["min_price"])
            except Exception:
                pass

        if "max_price" in filters and "price" in meta.columns:
            try:
                prices = pd.to_numeric(meta["price"].astype(str).str.replace("[^0-9.]+", ""), errors="coerce")
                mask &= prices <= float(filters["max_price"])
            except Exception:
                pass

        if "min_rating" in filters and "rating" in meta.columns:
            try:
                # rating may be textual like 'Three' or a star count; try to coerce
                ratings = pd.to_numeric(meta["rating"], errors="coerce")
                mask &= ratings >= float(filters["min_rating"])
            except Exception:
                pass

        if "genre" in filters and "genre" in meta.columns:
            genre = str(filters["genre"]).lower()
            has_genre = meta["genre"].fillna("").str.lower().str.contains(genre)
            mask &= has_genre.values

        # Set scores for masked-out items to a very low value so they won't be selected
        scores = np.where(mask, scores, -np.inf)

    if min_score is not None:
        scores = np.where(scores >= min_score, scores, -np.inf)

    best_idx = np.argsort(scores)[::-1][:top_k]

    # Filter out -inf results (no matches)
    best_idx = [i for i in best_idx if scores[i] != -np.inf]

    results = meta.iloc[best_idx].copy()
    results["score"] = scores[best_idx]
    return results


if __name__ == "__main__":
    query = input("Enter query: ")
    results = search(query)
    print(results.to_string(index=False))
