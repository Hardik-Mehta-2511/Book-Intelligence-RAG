# AI Book Intelligence Platform

This project scrapes the Books to Scrape catalog, cleans the data, and builds an AI-enabled book intelligence application.

## Project structure

- `scraper/` - web scraping and data cleaning scripts
- `data/raw/` - raw scraped CSV files
- `data/processed/` - cleaned data outputs
- `nlp/` - NLP and AI feature code
- `rag/` - retrieval-augmented generation code
- `app/` - Streamlit application and UI
- `excel/` - Excel analytics / Power Query support

## Get started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Scrape catalog-level data:
   ```bash
   python scraper/scraper.py
   ```

3. Scrape detailed book pages:
   ```bash
   python scraper/scraper2.py
   ```

4. Clean raw book data:
   ```bash
   python scraper/clean_data.py
   ```

5. Enrich descriptions and metadata:
   ```bash
   python nlp/enrich.py
   ```

6. Generate embeddings:
   ```bash
   python rag/embeddings.py
   ```

7. (Optional) Configure OpenAI for AI answers:
   - create a `.env` file with `OPENAI_API_KEY=your_key`

8. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

## Notes

- `books_raw.csv` contains the raw catalog-level data.
- `books_detailed.csv` will contain detailed product information.
- `books_clean.csv` is the cleaned dataset used for downstream AI and app development.
