import numpy as np
import pandas as pd
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = Path("data/processed/books_embeddings.npz")
META_FILE = Path("data/processed/books_embeddings_meta.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
_SBERT = None


def _coerce_numeric_price(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = text.replace("£", " ").replace("gbp", " ").replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _price_series(series):
    return pd.Series(series).map(_coerce_numeric_price)


def extract_price_filters(query: str) -> dict:
    """Extract explicit numeric price constraints from natural-language book queries."""
    if not isinstance(query, str) or not query.strip():
        return {}

    q = query.lower().strip()
    q = q.replace("£", " pounds ").replace("gbp", " pounds ")
    q = q.replace("€", " ").replace("$", " ")
    q = re.sub(r"\s+", " ", q)

    if not re.search(r"(under|below|less than|cheaper|over|above|more than|greater than|between|around|approximately|about|priced|costs?|costing|price)", q):
        return {}

    between_match = re.search(
        r"\bbetween\s+(?:pounds?\s+)?(\d+(?:\.\d+)?)\s*(?:and|to)\s+(?:pounds?\s+)?(\d+(?:\.\d+)?)\s*(?:pounds?)?",
        q,
    )
    if between_match:
        lower = float(between_match.group(1))
        upper = float(between_match.group(2))
        return {
            "min_price": min(lower, upper),
            "min_price_operator": "gte",
            "max_price": max(lower, upper),
            "max_price_operator": "lte",
        }

    under_match = re.search(
        r"\b(?:under|below|less than|cheaper than|not more than|costing less than)\s+(?:pounds?\s+)?(\d+(?:\.\d+)?)\s*(?:pounds?)?",
        q,
    )
    if under_match:
        return {"max_price": float(under_match.group(1)), "max_price_operator": "lt"}

    over_match = re.search(
        r"\b(?:over|above|more than|greater than|at least|costing more than)\s+(?:pounds?\s+)?(\d+(?:\.\d+)?)\s*(?:pounds?)?",
        q,
    )
    if over_match:
        return {"min_price": float(over_match.group(1)), "min_price_operator": "gt"}

    around_match = re.search(
        r"\b(?:around|approximately|about)\s+(?:pounds?\s+)?(\d+(?:\.\d+)?)\s*(?:pounds?)?",
        q,
    )
    if around_match:
        target = float(around_match.group(1))
        lower = target * 0.9
        upper = target * 1.1
        return {
            "min_price": lower,
            "min_price_operator": "gte",
            "max_price": upper,
            "max_price_operator": "lte",
        }

    exact_match = re.search(
        r"\b(?:priced at|price is|price was|exactly|costs?\s*(?:at)?|costing\s*(?:at)?)\s*(?:pounds?\s+)?(\d+(?:\.\d+)?)\s*(?:pounds?)?",
        q,
    )
    if exact_match:
        return {"exact_price": float(exact_match.group(1))}

    standalone_match = re.search(r"(?:pounds?\s+)?(\d+(?:\.\d+)?)\s*(?:pounds?)?", q)
    if standalone_match and re.search(r"\b(?:price|cost|pounds)\b", q):
        return {"exact_price": float(standalone_match.group(1))}

    return {}


def apply_price_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if df is None or df.empty or not filters:
        return df.copy() if df is not None else df

    result = df.copy()
    if "price" not in result.columns and "price_clean" not in result.columns:
        return result

    price_col = "price_clean" if "price_clean" in result.columns else "price"
    prices = _price_series(result[price_col])
    mask = pd.Series(True, index=result.index)

    if "min_price" in filters:
        op = filters.get("min_price_operator", "gte")
        if op == "gt":
            mask &= prices > float(filters["min_price"])
        else:
            mask &= prices >= float(filters["min_price"])

    if "max_price" in filters:
        op = filters.get("max_price_operator", "lte")
        if op == "lt":
            mask &= prices < float(filters["max_price"])
        else:
            mask &= prices <= float(filters["max_price"])

    if "exact_price" in filters:
        mask &= np.isclose(prices, float(filters["exact_price"]), rtol=0, atol=1e-6)

    return result[mask].copy()


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

    query_filters = extract_price_filters(query)
    combined_filters = dict(filters or {})
    combined_filters.update(query_filters)

    query_vec = embed_query(query)
    scores = cosine_similarity(embeddings, query_vec)

    if combined_filters:
        mask = np.ones(len(scores), dtype=bool)

        if "min_price" in combined_filters and "price" in meta.columns:
            try:
                prices = _price_series(meta["price"])
                op = combined_filters.get("min_price_operator", "gte")
                if op == "gt":
                    mask &= prices.to_numpy() > float(combined_filters["min_price"])
                else:
                    mask &= prices.to_numpy() >= float(combined_filters["min_price"])
            except Exception:
                pass

        if "max_price" in combined_filters and "price" in meta.columns:
            try:
                prices = _price_series(meta["price"])
                op = combined_filters.get("max_price_operator", "lte")
                if op == "lt":
                    mask &= prices.to_numpy() < float(combined_filters["max_price"])
                else:
                    mask &= prices.to_numpy() <= float(combined_filters["max_price"])
            except Exception:
                pass

        if "exact_price" in combined_filters and "price" in meta.columns:
            try:
                prices = _price_series(meta["price"])
                mask &= np.isclose(prices.to_numpy(), float(combined_filters["exact_price"]), rtol=0, atol=1e-6)
            except Exception:
                pass

        if "min_rating" in combined_filters and "rating" in meta.columns:
            try:
                ratings = pd.to_numeric(meta["rating"], errors="coerce")
                mask &= ratings.to_numpy() >= float(combined_filters["min_rating"])
            except Exception:
                pass

        if "genre" in combined_filters and "genre" in meta.columns:
            genre = str(combined_filters["genre"]).lower()
            has_genre = meta["genre"].fillna("").str.lower().str.contains(genre)
            mask &= has_genre.to_numpy()

        scores = np.where(mask, scores, -np.inf)

    if min_score is not None:
        scores = np.where(scores >= min_score, scores, -np.inf)

    best_idx = np.argsort(scores)[::-1][:top_k]
    best_idx = [i for i in best_idx if scores[i] != -np.inf]

    results = meta.iloc[best_idx].copy()
    results["score"] = scores[best_idx]
    return results


if __name__ == "__main__":
    query = input("Enter query: ")
    results = search(query)
    print(results.to_string(index=False))
