import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("data/processed/books_enriched.csv")
OUTPUT_FILE = Path("data/processed/books_embeddings.npz")
META_FILE = Path("data/processed/books_embeddings_meta.csv")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Enriched file not found: {INPUT_FILE}")
    return pd.read_csv(INPUT_FILE, encoding="utf-8")


def main() -> None:
    df = load_data()
    model = SentenceTransformer(MODEL_NAME)

    descriptions = df["description"].fillna("").astype(str).tolist()
    embeddings = model.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, embeddings=embeddings)
    df[["title", "url", "genre", "themes", "audience", "keywords", "summary", "description", "price", "rating", "availability"]].to_csv(META_FILE, index=False, encoding="utf-8")

    print(f"Saved embeddings to: {OUTPUT_FILE}")
    print(f"Saved metadata to: {META_FILE}")


if __name__ == "__main__":
    main()
