import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import traceback
import pandas as pd
import re
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()


def get_openai_api_key():
    value = os.getenv("OPENAI_API_KEY")
    if value:
        return value

    try:
        secret = st.secrets.get("OPENAI_API_KEY")
        if secret:
            return str(secret)
        secret = st.secrets.get("openai", {}).get("api_key")
        if secret:
            return str(secret)
    except Exception:
        pass

    return None

DATA_FILE = "data/processed/books_enriched.csv"
EMBEDDINGS_FILE = Path("data/processed/books_embeddings.npz")

RETRIEVER_IMPORT_ERROR = None
OPENAI_IMPORT_ERROR = None
try:
    from rag.retriever import search
except Exception:
    search = None
    RETRIEVER_IMPORT_ERROR = traceback.format_exc()

try:
    from rag.openai_rag import answer_query
    OPENAI_AVAILABLE = True
except Exception:
    answer_query = None
    OPENAI_AVAILABLE = False
    OPENAI_IMPORT_ERROR = traceback.format_exc()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, encoding="utf-8")

    # Ensure numeric price column exists
    if "price_clean" not in df.columns:
        if "price" in df.columns:
            def parse_price(p):
                try:
                    if pd.isna(p):
                        return None
                    s = str(p)
                    s = re.sub(r"[^0-9.]+", "", s)
                    return float(s) if s != "" else None
                except Exception:
                    return None

            df["price_clean"] = df["price"].apply(parse_price)
        else:
            df["price_clean"] = None

    # Ensure numeric rating column exists (map words to ints)
    if "rating_clean" not in df.columns:
        mapping = {"Zero": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
        if "rating" in df.columns:
            def parse_rating(r):
                try:
                    if pd.isna(r):
                        return 0
                    # Some datasets store words like 'Three' or numerics
                    if isinstance(r, (int, float)):
                        return int(r)
                    rs = str(r).strip()
                    if rs.isdigit():
                        return int(rs)
                    return mapping.get(rs, 0)
                except Exception:
                    return 0

            df["rating_clean"] = df["rating"].apply(parse_rating)
        else:
            df["rating_clean"] = 0

    # Availability simplified
    if "availability_clean" not in df.columns:
        src = None
        if "availability_detail" in df.columns:
            src = df["availability_detail"].astype(str)
        elif "availability" in df.columns:
            src = df["availability"].astype(str)

        if src is not None:
            def parse_avail(a):
                try:
                    if pd.isna(a):
                        return 0
                    s = str(a)
                    m = re.search(r"(\d+)", s)
                    if m:
                        return int(m.group(1))
                    return 1 if "in stock" in s.lower() else 0
                except Exception:
                    return 0

            df["availability_clean"] = src.apply(parse_avail)
        else:
            df["availability_clean"] = 0

    return df


def format_price(value):
    try:
        return f"£{float(value):.2f}"
    except (ValueError, TypeError):
        return ""


def main():
    st.set_page_config(page_title="AI Book Intelligence", layout="wide")

    st.title("📚 AI Book Intelligence")
    st.markdown(
        "Search the Books to Scrape catalog, inspect individual books, and explore the cleaned dataset."
    )

    df = load_data()

    # Inject custom CSS for a cleaner, themed UI and AI answer box at the top
    st.markdown(
        """
    <style>
    :root{
      --bg:#f6f5f2; /* very light cream */
      --panel:#ffffff; /* white panels */
      --accent:#2b8a7a; /* teal accent */
      --muted:#2f3b3a; /* muted text */
      --muted-2:#6b7a78;
      --card:#ffffff; /* card background */
    }

    /* Page background and global font */
    .stApp { background: var(--bg); color: var(--muted); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }

    /* Main container spacing */
    .main .block-container { max-width: 1400px; padding: 2rem 2.5rem; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid rgba(99,110,108,0.06); }
    section[data-testid="stSidebar"] > div { padding: 1.2rem; }

    /* AI box / hero */
    .ai-box{ background: var(--card); border: 1px solid rgba(43,138,122,0.08); border-radius:14px; padding:18px; box-shadow: 0 6px 18px rgba(30,40,40,0.04); }
    .ai-answer{ background: #fbfbfb; color: var(--muted); border-radius:10px; padding:16px; margin-top:12px; font-size:15px; line-height:1.6 }
    .ai-answer h4{ margin:0 0 8px 0; font-family: 'Georgia', serif; color: #16332b }

    /* Buttons and inputs */
    div.stButton > button { background: var(--accent); color: white; border-radius:10px; padding:10px 18px; font-weight:600; }
    div.stButton > button:hover { filter:brightness(0.95); }
    input, textarea, .stTextInput>div>input { font-size:15px !important; color: var(--muted); }

    /* Cards and book tiles */
    .book-card { background: var(--panel); border: 1px solid rgba(99,110,108,0.06); border-radius:12px; box-shadow: 0 6px 18px rgba(20,30,30,0.03); padding:12px; }
    .book-title { color: #16332b; font-weight:700; font-size:15px }
    .book-price { color: var(--accent); font-weight:700 }

    /* Footer */
    .footer { background: transparent; color: var(--muted-2); }

    /* Make sure text contrast is good */
    .stMarkdown, .stText { color: var(--muted); }

    </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar filters (moved above AI box so values are available immediately)
    st.sidebar.header("Filters")
    search_text = st.sidebar.text_input("Search title or description")
    min_price = st.sidebar.number_input("Min price", min_value=0.0, value=0.0, step=1.0)
    max_price = st.sidebar.number_input("Max price", min_value=0.0, value=float(df["price_clean"].max()), step=1.0)
    min_rating = st.sidebar.selectbox(
        "Min rating", [0, 1, 2, 3, 4, 5], index=0
    )

    # Top AI box: prompt + answer area (button placed below the prompt)
    ai_container = st.container()
    with ai_container:
        query_top = st.text_area("Ask anything about books...", key="ai_top_input", placeholder="e.g., Recommend adventure books under £20", height=140)

        # Generate button placed below the prompt for clearer flow
        ask_top = st.button("Generate answer", key="ai_top_button")

        # Answer placeholder container (empty until generation)
        answer_holder = st.empty()

        # Example prompts dropdown
        examples = [
            "Find me adventure books about friendship under £25",
            "Recommend mystery novels with female protagonists",
            "Summarize 'A Light in the Attic'",
            "Which books are good for young readers about bravery?",
        ]
        chosen = st.selectbox("Try an example", ["-- pick an example --"] + examples, key="ai_example_select")

        # Initialize search history in session state
        if "search_history" not in st.session_state:
            st.session_state.search_history = []

        # Build filters dict for retriever using sidebar values
        filters = {}
        try:
            if max_price and max_price > 0:
                filters["max_price"] = max_price
            if min_rating and min_rating > 0:
                filters["min_rating"] = min_rating
        except Exception:
            pass

        typed_query = (query_top or "").strip()
        selected_example = chosen if chosen and chosen != "-- pick an example --" else ""
        query_to_execute = typed_query if typed_query else selected_example

        # Placeholder for results and retrieval — reuse existing logic
        if ask_top:
            if not query_to_execute:
                st.warning("Please enter a query or select an example before generating an answer.")
            else:
                with st.spinner("Retrieving relevant books and generating answer…"):
                    results = []
                    if EMBEDDINGS_FILE.exists() and search is not None:
                        try:
                            results = search(query_to_execute, top_k=5, filters=filters)
                            # push into search history (most recent first)
                            st.session_state.search_history.insert(0, query_to_execute)
                            # keep history bounded
                            st.session_state.search_history = st.session_state.search_history[:20]
                        except Exception as exc:
                            st.error(f"Retriever failed: {exc}")
                            results = []
                    else:
                        if not EMBEDDINGS_FILE.exists():
                            st.warning("Embeddings not found. Run `python rag/embeddings.py` first.")
                        else:
                            st.warning("Retriever not available (import error). Check `rag/retriever.py`.")
                            if RETRIEVER_IMPORT_ERROR:
                                st.code(RETRIEVER_IMPORT_ERROR, language="text")

                # show retrieval + AI answer
                if len(results):
                    st.markdown("### Top matching books")
                    # show cards with covers if available
                    img_col = None
                    for candidate in ("image_url", "thumbnail", "cover_url"):
                        if candidate in results.columns:
                            img_col = candidate
                            break

                    cols = st.columns(min(5, len(results)))
                    for i, (_, r) in enumerate(results.iterrows()):
                        with cols[i]:
                            st.markdown('<div class="book-card">', unsafe_allow_html=True)
                            if img_col and pd.notna(r.get(img_col, None)) and r.get(img_col):
                                try:
                                    st.image(r.get(img_col), use_column_width=True)
                                except Exception:
                                    pass
                            st.markdown(f"<div class=\"book-content\">\n<div class=\"book-title\">{r.get('title','')}</div>\n<div class=\"book-category\">{r.get('genre','')}</div>\n<div class=\"book-price\">{format_price(r.get('price') or r.get('price_clean',''))}</div>\n</div>", unsafe_allow_html=True)
                            # simple local explanation
                            def explain_match(q, row):
                                q_words = set([w.lower() for w in re.findall(r"\w+", q) if len(w) > 3])
                                desc = str(row.get('description','') or '')
                                desc_words = set([w.lower() for w in re.findall(r"\w+", desc) if len(w) > 3])
                                overlap = q_words & desc_words
                                reasons = []
                                if overlap:
                                    reasons.append(f"Contains keywords: {', '.join(list(overlap)[:6])}.")
                                if row.get('genre'):
                                    reasons.append(f"Genre: {row.get('genre')}")
                                reasons.append(f"Semantic match score: {row.get('score',0):.2f}")
                                return ' '.join(reasons)

                            with st.expander("Why this book?"):
                                st.write(explain_match(query_to_execute, r))

                            # link to detail view
                            book_key = f"book_{i}"
                            if st.button("View details", key=book_key):
                                st.session_state.selected_book = r.to_dict()
                            st.markdown('</div>', unsafe_allow_html=True)

                    if OPENAI_AVAILABLE and get_openai_api_key():
                        try:
                            answer = answer_query(query_to_execute, top_k=5)
                            answer_holder.markdown(f"<div class='ai-answer'><h4>AI Recommendation</h4><div>{answer}</div></div>", unsafe_allow_html=True)
                        except Exception as exc:
                            st.error(f"OpenAI generation failed: {exc}")
                    else:
                        st.info("OpenAI is not configured. Set OPENAI_API_KEY in your local .env or in Streamlit secrets to enable AI answers.")
                        if OPENAI_IMPORT_ERROR:
                            st.code(OPENAI_IMPORT_ERROR, language="text")
                        else:
                            st.write("Hint: create a `.env` file with `OPENAI_API_KEY=sk-...` or add the key in Streamlit Cloud secrets.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Search history
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Recent searches")
    history = st.session_state.get("search_history", [])
    if history:
        for i, h in enumerate(history[:8]):
            if st.sidebar.button(h, key=f"hist_{i}"):
                # set top input when clicked
                st.experimental_set_query_params()  # no-op placeholder to keep session consistent
                st.session_state.ai_top_input = h
    else:
        st.sidebar.markdown("No recent searches")

    st.sidebar.markdown("---")
    with st.sidebar.expander("How it works", expanded=False):
        st.markdown(
            """
            This app uses semantic embeddings to find books similar to your query. We embed both the query and the book descriptions with a SentenceTransformers model, compute cosine similarity, then rank results. An LLM (OpenAI) composes a concise answer using the retrieved book contexts (RAG).
            """
        )

    filtered = df.copy()

    if search_text:
        filtered = filtered[filtered["title"].str.contains(search_text, case=False, na=False) |
                            filtered["description"].str.contains(search_text, case=False, na=False)]

    if max_price > 0:
        filtered = filtered[(filtered["price_clean"] >= min_price) & (filtered["price_clean"] <= max_price)]

    if min_rating > 0:
        filtered = filtered[filtered["rating_clean"] >= min_rating]

    st.subheader("Catalog overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Books", len(df))
    col2.metric("Filtered", len(filtered))
    col3.metric("Average price", format_price(filtered["price_clean"].mean() if len(filtered) else 0))

    st.write("### Filtered books")
    st.dataframe(
        filtered[["title", "price_clean", "rating_clean", "availability_clean", "url"]].rename(
            columns={
                "price_clean": "price",
                "rating_clean": "rating",
                "availability_clean": "available"
            }
        ),
        use_container_width=True,
    )

    if len(filtered) > 0:
        selected_index = st.number_input(
            "Select book row", min_value=1, max_value=len(filtered), value=1, step=1
        )
        selected_book = filtered.iloc[selected_index - 1]

        st.write("### Book details")
        st.markdown(f"**Title:** {selected_book['title']}")
        st.markdown(f"**Price:** {format_price(selected_book['price_clean'])}")
        st.markdown(f"**Rating:** {selected_book.get('rating_clean', '')} stars")
        st.markdown(f"**Availability:** {selected_book.get('availability_detail', '')}")
        st.markdown(f"**Genre:** {selected_book.get('genre', '')}")
        st.markdown(f"**Themes:** {selected_book.get('themes', '')}")
        st.markdown(f"**Audience:** {selected_book.get('audience', '')}")
        st.markdown(f"**Summary:** {selected_book.get('summary', '')}")
        st.markdown(f"**Description:** {selected_book.get('description', '')}")
        st.markdown(f"**URL:** {selected_book.get('url', '')}")

    # AI prompt moved to the top of the page. Legacy bottom AI UI removed.


if __name__ == "__main__":
    main()
