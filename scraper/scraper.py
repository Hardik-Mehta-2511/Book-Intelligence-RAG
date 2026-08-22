import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

book_data = []

for page in range(1, 51):

    if page == 1:
        url = BASE_URL + "index.html"
    else:
        url = BASE_URL + f"catalogue/page-{page}.html"

    print(f"Scraping page {page}...")

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.encoding = response.apparent_encoding or "utf-8"

    print("Status:", response.status_code)

    soup = BeautifulSoup(response.text, "html.parser")

    books = soup.select("article.product_pod")

    print(f"Books found: {len(books)}")

    for book in books:

        title = book.h3.a["title"]

        price = book.select_one(
            ".price_color"
        ).get_text(strip=True).replace("Â", "")

        rating = book.select_one(
            "p.star-rating"
        )["class"][1]

        availability = book.select_one(
            ".availability"
        ).get_text(" ", strip=True)

        relative_url = book.h3.a["href"]

        book_url = urljoin(url, relative_url)


        book_data.append({
            "title": title,
            "price": price,
            "rating": rating,
            "availability": availability,
            "url": book_url
        })

df = pd.DataFrame(book_data)

print("\n-----------------------------")
print("Total books scraped:", len(df))
print("-----------------------------")

output_path = "data/raw/books_raw.csv"

df.to_csv(
    output_path,
    index=False
)

print(f"Saved to: {output_path}")