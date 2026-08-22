import pandas as pd
import re
from pathlib import Path

RAW_FILE = Path("data/raw/books_raw.csv")
DETAILED_FILE = Path("data/raw/books_detailed.csv")
OUTPUT_FILE = Path("data/processed/books_clean.csv")

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def clean_price(price_value: str) -> float:
    if not isinstance(price_value, str):
        return None
    cleaned = re.sub(r"[^0-9\.]+", "", price_value)
    return float(cleaned) if cleaned else None


def clean_availability(value: str) -> int:
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else 0


def read_source() -> pd.DataFrame:
    if DETAILED_FILE.exists():
        return pd.read_csv(DETAILED_FILE, encoding="utf-8")
    if RAW_FILE.exists():
        return pd.read_csv(RAW_FILE, encoding="utf-8")
    raise FileNotFoundError("No raw or detailed source file found.")


def main() -> None:
    df = read_source()

    df["price"] = df["price"].astype(str).str.replace("Â", "", regex=False)
    df["price_clean"] = df["price"].apply(clean_price)
    df["rating_clean"] = df["rating"].map(rating_map)
    df["availability_clean"] = df["availability"].apply(clean_availability)

    if "price_excl_tax" in df.columns:
        df["price_excl_tax_clean"] = df["price_excl_tax"].astype(str).apply(clean_price)
    if "price_incl_tax" in df.columns:
        df["price_incl_tax_clean"] = df["price_incl_tax"].astype(str).apply(clean_price)
    if "tax" in df.columns:
        df["tax_clean"] = df["tax"].astype(str).apply(clean_price)
    if "number_of_reviews" in df.columns:
        df["number_of_reviews"] = pd.to_numeric(df["number_of_reviews"], errors="coerce").fillna(0).astype(int)

    df["title"] = df["title"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["description"] = df.get("description", pd.Series([""] * len(df))).astype(str).str.strip()

    df = df.dropna(subset=["title", "url"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Saved cleaned dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
