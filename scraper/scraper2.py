import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

INPUT_FILE = "data/raw/books_raw.csv"
OUTPUT_FILE = "data/raw/books_detailed.csv"

headers = {
    "User-Agent": "Mozilla/5.0"
}

df = pd.read_csv(INPUT_FILE)

detailed_data = []

for index, row in df.iterrows():

    url = row["url"]

    print(f"Scraping {index + 1}/{len(df)}: {row['title']}")

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.encoding = response.apparent_encoding or "utf-8"

        if response.status_code != 200:

            print(
                f"  Failed: {response.status_code}"
            )

            continue

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Description

        description_tag = soup.select_one(
            "#product_description + p"
        )

        description = (
            description_tag.get_text(
                " ",
                strip=True
            )
            if description_tag
            else ""
        )

        # Product information

        product_info = {}

        table = soup.select_one(
            "table.table-striped"
        )

        if table:

            for table_row in table.select("tr"):

                cells = table_row.select(
                    "th, td"
                )

                if len(cells) == 2:

                    key = cells[0].get_text(
                        strip=True
                    )

                    value = cells[1].get_text(
                        " ",
                        strip=True
                    )

                    product_info[key] = value

        detailed_data.append({

            "title": row["title"],

            "price": row["price"],

            "rating": row["rating"],

            "availability": row["availability"],

            "url": url,

            "upc": product_info.get(
                "UPC", ""
            ),

            "product_type": product_info.get(
                "Product Type", ""
            ),

            "price_excl_tax": product_info.get(
                "Price (excl. tax)", ""
            ),

            "price_incl_tax": product_info.get(
                "Price (incl. tax)", ""
            ),

            "tax": product_info.get(
                "Tax", ""
            ),

            "availability_detail": product_info.get(
                "Availability", ""
            ),

            "number_of_reviews": product_info.get(
                "Number of reviews", ""
            ),

            "description": description

        })

        time.sleep(0.2)

    except requests.exceptions.RequestException as e:

        print(
            f"  Request failed: {e}"
        )


detailed_df = pd.DataFrame(
    detailed_data
)

detailed_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\n-----------------------------")
print(
    "Detailed books scraped:",
    len(detailed_df)
)
print("-----------------------------")

print(
    f"Saved to: {OUTPUT_FILE}"
)