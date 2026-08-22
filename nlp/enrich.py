import pandas as pd
import re
from pathlib import Path

CLEAN_FILE = Path("data/processed/books_clean.csv")
DETAILED_FILE = Path("data/raw/books_detailed.csv")
OUTPUT_FILE = Path("data/processed/books_enriched.csv")

GENRE_KEYWORDS = {
    "Fantasy": ["magic", "dragon", "kingdom", "wizard", "sorcerer", "fairy"],
    "Science Fiction": ["space", "alien", "future", "robot", "galaxy", "science fiction"],
    "Mystery": ["murder", "detective", "mystery", "investigation", "crime", "case"],
    "Romance": ["love", "romance", "relationship", "heart", "wedding", "affair"],
    "Horror": ["ghost", "haunted", "horror", "blood", "monster", "fear"],
    "Historical": ["history", "historical", "period", "Victorian", "war", "empire"],
    "Biography": ["memoir", "biography", "autobiography", "life of", "story of"],
    "Nonfiction": ["nonfiction", "guide", "reference", "essay", "true story", "history"],
    "Children": ["child", "children", "kid", "young reader", "family"],
}

THEME_KEYWORDS = {
    "Adventure": ["adventure", "journey", "quest", "exploration", "escape"],
    "Friendship": ["friendship", "friends", "companionship", "allies"],
    "Courage": ["courage", "bravery", "hero", "fearless", "bold"],
    "Love": ["love", "romance", "heart", "passion", "affection"],
    "Coming of Age": ["growing up", "coming of age", "teen", "young adult", "mature"],
    "Mystery": ["mystery", "secret", "investigation", "suspense", "clues"],
    "Family": ["family", "mother", "father", "children", "siblings"],
    "Humor": ["humor", "funny", "comedy", "laugh", "satire"],
    "Suspense": ["suspense", "tension", "thriller", "danger", "uncertainty"],
}

AUDIENCE_KEYWORDS = {
    "Children": ["children", "kids", "young readers", "juvenile", "picture book"],
    "Teens": ["teen", "young adult", "YA", "high school", "coming of age"],
    "Adults": ["adult", "mature", "complex", "novel for adults", "grown-ups"],
    "Families": ["family", "parents", "sibling", "home", "household"],
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^\w\s']", " ", text).lower().strip()


def first_sentences(text: str, count: int = 2) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:count]).strip()


def extract_labels(text: str, keyword_map: dict, max_labels: int = 3) -> str:
    normalized = normalize_text(text)
    labels = []
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword.lower() in normalized:
                labels.append(label)
                break
        if len(labels) >= max_labels:
            break
    return "; ".join(labels)


def extract_keywords(text: str, max_keywords: int = 5) -> str:
    normalized = normalize_text(text)
    words = [word for word in normalized.split() if len(word) > 3]
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    sorted_words = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    keywords = [word for word, _ in sorted_words[:max_keywords]]
    return "; ".join(keywords)


def load_input_file() -> Path:
    if DETAILED_FILE.exists():
        return DETAILED_FILE
    if CLEAN_FILE.exists():
        return CLEAN_FILE
    raise FileNotFoundError("No books_clean.csv or books_detailed.csv found for enrichment.")


def main() -> None:
    input_file = load_input_file()
    df = pd.read_csv(input_file, encoding="utf-8")

    if "description" not in df.columns:
        df["description"] = ""

    df["description"] = df["description"].fillna("")

    df["summary"] = df["description"].apply(lambda text: first_sentences(text, count=2))
    df["genre"] = df["description"].apply(lambda text: extract_labels(text, GENRE_KEYWORDS))
    df["themes"] = df["description"].apply(lambda text: extract_labels(text, THEME_KEYWORDS))
    df["audience"] = df["description"].apply(lambda text: extract_labels(text, AUDIENCE_KEYWORDS))
    df["keywords"] = df["description"].apply(lambda text: extract_keywords(text, max_keywords=5))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Saved enriched dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
