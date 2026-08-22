import os
from pathlib import Path

from openai import OpenAI
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import time
import re
from typing import Optional, Dict, Any

# Local retriever
from rag.retriever import search as retriever_search

EMBEDDINGS_FILE = Path("data/processed/books_embeddings.npz")
META_FILE = Path("data/processed/books_embeddings_meta.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
# Cache SBERT model to avoid reloading on every call
_SBERT = None
# OpenAI model can be set via environment variable `OPENAI_MODEL`.
# Default to `gpt-5-mini` to prefer GPT-5 family models when available.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")


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


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_title_matches(query: str, meta: pd.DataFrame) -> pd.DataFrame:
    """Return rows from meta whose titles match the query by a few heuristics.

    Heuristics:
    - title substring appears in query (case-insensitive)
    - query substring appears in title
    - all nontrivial words from title appear in the query (set subset)
    """
    qn = _normalize_text(query)
    q_words = set([w for w in qn.split() if len(w) > 2])
    matches = []
    for _, row in meta.iterrows():
        title = row.get("title", "")
        tn = _normalize_text(title)
        if not tn:
            continue
        # direct contains checks
        if tn in qn or qn in tn:
            matches.append(row)
            continue
        # word-subset check (require at least 2 significant words)
        t_words = set([w for w in tn.split() if len(w) > 2])
        if len(t_words) >= 2 and t_words.issubset(q_words):
            matches.append(row)
            continue

    if matches:
        return pd.DataFrame(matches)
    return pd.DataFrame()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)


def retrieve_docs(query: str, top_k: int = 5):
    embeddings = load_embeddings()
    meta = load_meta()
    query_vec = embed_query(query)
    scores = cosine_similarity(embeddings, query_vec)
    best_idx = np.argsort(scores)[::-1][:top_k]
    results = meta.iloc[best_idx].copy()
    results["score"] = scores[best_idx]
    return results


def build_prompt(query: str, docs: pd.DataFrame) -> str:
    context_items = []
    for _, row in docs.iterrows():
        context_items.append(
            f"Title: {row['title']}\nGenre: {row.get('genre', '')}\nThemes: {row.get('themes', '')}\nAudience: {row.get('audience', '')}\nPrice: {row.get('price', '')}\nRating: {row.get('rating', '')}\nDescription: {row.get('description', '')}\n"
        )

    context = "\n---\n".join(context_items)
    return (
        f"You are an AI assistant for a book catalog. Use the provided book context to answer the user query. "
        f"If the answer is not in the provided books, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"User query: {query}\n\n"
        f"Answer:" 
    )


def answer_query(query: str, top_k: int = 5) -> str:
    """Answer a user query by combining retrieved docs and the LLM.

    Behavior:
    - Prepend title-fallback rows to retrieved docs when applicable.
    - If no catalog context is available, the LLM will still answer from
      general knowledge but will clearly note the answer is not drawn
      from the catalog and may be unverified.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured. Set it in your environment or .env file.")

    # semantic retrieval
    docs = retrieve_docs(query, top_k=top_k)

    # prepend title-fallback matches if present
    try:
        meta = load_meta()
        title_matches = find_title_matches(query, meta)
        if not title_matches.empty:
            title_matches = title_matches.copy()
            title_matches["score"] = 1.0
            combined = pd.concat([title_matches, docs], ignore_index=True)
            if "url" in combined.columns:
                combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
            docs = combined.iloc[:top_k].copy()
    except Exception:
        pass

    # Build a clear prompt that asks the model to prefer catalog context but fall
    # back to general knowledge if needed (and to declare when it does so).
    context_items = []
    for i, row in enumerate(docs.itertuples(), start=1):
        context_items.append(
            f"[{i}] Title: {getattr(row, 'title', '')}\nURL: {getattr(row, 'url', '')}\nScore: {getattr(row, 'score', 0.0):.3f}\nDescription: {getattr(row, 'description', '')}\n"
        )

    context = "\n---\n".join(context_items) if context_items else ""

    system_msg = (
        "You are a helpful, high-quality assistant specialized in books.\n"
        "When possible, use only the provided catalog contexts to answer and cite them by index (e.g., [1]).\n"
        "If the catalog contexts do not contain the needed information, answer from general knowledge, and begin your reply with the sentence:\n"
        "'Note: the following answer is based on general knowledge because the catalog does not contain this information.'\n"
        "Keep answers concise (2-6 sentences) and include citations for any fact taken from the contexts."
    )

    user_msg = f"Context:\n{context}\n\nUser query: {query}\n\nAnswer:" if context else f"User query: {query}\n\nAnswer:" 

    client = OpenAI(api_key=openai_api_key)
    model_name = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_completion_tokens=450,
    )

    return resp.choices[0].message.content.strip()


def enrich_description(description: str, max_retries: int = 3) -> dict:
    """Call the LLM to produce a structured enrichment for a book description.

    Returns a dict with keys: summary, themes, genre, audience, keywords
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    system_msg = (
        "You are an assistant that extracts structured metadata from a book description. "
        "Return a single JSON object with keys: summary, themes (list), genre, audience, keywords (list)."
    )

    user_prompt = (
        f"Description:\n{description}\n\n"
        "Respond ONLY with valid JSON. Do not include any explanatory text."
    )

    for attempt in range(1, max_retries + 1):
        try:
            client = OpenAI(api_key=openai_api_key)
            model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=400,
            )

            content = resp.choices[0].message.content.strip()

            # Parse JSON — some models may wrap output in backticks or code blocks
            try:
                # Trim code fence if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    if content.endswith("```"):
                        content = content.rsplit("\n", 1)[0]

                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError:
                # Attempt to recover by finding first/last brace
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    try:
                        parsed = json.loads(content[start : end + 1])
                        return parsed
                    except json.JSONDecodeError:
                        pass

                raise

        except Exception as exc:
            if attempt == max_retries:
                raise
            sleep_time = 2 ** attempt
            time.sleep(sleep_time)


def batch_enrich(input_meta_csv: str = "data/processed/books_clean.csv", output_csv: str = "data/processed/books_enriched.csv", start: int = 0, end: int = None):
    """Load cleaned book CSV, enrich descriptions via the LLM, and save an enriched CSV.

    This function is safe to run when `OPENAI_API_KEY` is configured. It will append
    structured fields returned by `enrich_description` to the dataframe.
    """
    df = pd.read_csv(input_meta_csv, encoding="utf-8")
    if end is None:
        end = len(df)

    enriched = []
    for idx, row in df.iloc[start:end].iterrows():
        desc = row.get("description", "") or ""
        try:
            meta = enrich_description(desc)
        except Exception as exc:
            meta = {"summary": "", "themes": [], "genre": "", "audience": "", "keywords": []}

        out_row = row.to_dict()
        out_row.update({
            "ai_summary": meta.get("summary", ""),
            "ai_genre": meta.get("genre", ""),
            "ai_themes": json.dumps(meta.get("themes", []), ensure_ascii=False),
            "ai_audience": meta.get("audience", ""),
            "ai_keywords": json.dumps(meta.get("keywords", []), ensure_ascii=False),
        })
        enriched.append(out_row)

    out_df = pd.DataFrame(enriched)
    out_df.to_csv(output_csv, index=False)
    return out_df


def rag_answer(query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None, min_score: Optional[float] = None, allow_fallback: bool = True) -> str:
    """Retrieve top documents and ask the LLM to answer using only those sources.

    Returns the assistant's text reply.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    docs = retriever_search(query, top_k=top_k, filters=filters or {}, min_score=min_score)

    # If retriever returned nothing, try a title-match fallback: look for
    # any exact/substring matches in the metadata titles and include them.
    if docs.empty:
        meta = load_meta()
        title_matches = meta[meta["title"].str.contains(query, case=False, na=False)]
        if not title_matches.empty:
            title_matches = title_matches.copy()
            title_matches["score"] = 1.0
            docs = title_matches.iloc[:top_k].copy()
    else:
        # Also prepend any strong title-match results so explicit title queries
        # are always included in the RAG context.
        try:
            meta = load_meta()
            title_matches = meta[meta["title"].str.contains(query, case=False, na=False)]
            if not title_matches.empty:
                title_matches = title_matches.copy()
                title_matches["score"] = 1.0
                # combine while preserving order and deduplicating by URL
                combined = pd.concat([title_matches, docs], ignore_index=True)
                if "url" in combined.columns:
                    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
                docs = combined.iloc[:top_k].copy()
        except Exception:
            # if metadata lookup fails for any reason, continue with original docs
            pass

    # If no docs and fallback disabled, return explicit message.
    if docs.empty and not allow_fallback:
        return "No relevant documents found in the catalog."

    # Build context with numbered citations
    context_parts = []
    for i, row in enumerate(docs.itertuples(), start=1):
        title = getattr(row, "title", "")
        url = getattr(row, "url", "")
        desc = getattr(row, "description", "")
        score = getattr(row, "score", 0.0)
        context_parts.append(f"[{i}] Title: {title}\nURL: {url}\nScore: {score:.3f}\nDescription: {desc}\n")

    context = "\n---\n".join(context_parts)

    # Build prompt that prefers catalog context but can fallback to general knowledge
    system_msg = (
        "You are a helpful AI assistant focused on books.\n"
        "Prefer answers that use the provided catalog contexts and cite them by index (e.g., [1]).\n"
        "If the catalog does not contain the information, answer from general knowledge but start the reply with:\n"
        "'Note: the following answer is based on general knowledge because the catalog does not contain this information.'\n"
        "Keep answers concise (2-6 sentences) and include citations for any context-based facts."
    )

    user_msg = (
        f"Context:\n{context}\n\nUser question: {query}\n\nProvide a concise answer (2-6 sentences) and include citations where appropriate."
        if context
        else f"User question: {query}\n\nProvide a concise answer (2-6 sentences)."
    )

    client = OpenAI(api_key=openai_api_key)
    model_name = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_completion_tokens=450,
    )

    return resp.choices[0].message.content.strip()
