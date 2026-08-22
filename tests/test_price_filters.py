import pandas as pd

from rag.retriever import apply_price_filters, extract_price_filters


def test_parse_under_price_variants():
    cases = {
        "books under £25": {"max_price": 25.0, "max_price_operator": "lt"},
        "books under 25 pounds": {"max_price": 25.0, "max_price_operator": "lt"},
        "books below 25": {"max_price": 25.0, "max_price_operator": "lt"},
        "books cheaper than £25": {"max_price": 25.0, "max_price_operator": "lt"},
        "books over £40": {"min_price": 40.0, "min_price_operator": "gt"},
        "books above 40 pounds": {"min_price": 40.0, "min_price_operator": "gt"},
        "books between £20 and £30": {"min_price": 20.0, "min_price_operator": "gte", "max_price": 30.0, "max_price_operator": "lte"},
        "books between 20 and 30 pounds": {"min_price": 20.0, "min_price_operator": "gte", "max_price": 30.0, "max_price_operator": "lte"},
        "books priced at £25": {"exact_price": 25.0},
        "books under £25 with rating 4 or higher": {"max_price": 25.0, "max_price_operator": "lt"},
    }

    for query, expected in cases.items():
        actual = extract_price_filters(query)
        assert actual == expected, f"{query} -> {actual!r} != {expected!r}"


def test_apply_price_filters_keeps_only_matches():
    df = pd.DataFrame({"price": [10.0, 22.5, 24.99, 25.0, 26.0, 40.0]})

    under = apply_price_filters(df, {"max_price": 25.0, "max_price_operator": "lt"})
    assert under["price"].tolist() == [10.0, 22.5, 24.99]

    over = apply_price_filters(df, {"min_price": 40.0, "min_price_operator": "gt"})
    assert over["price"].tolist() == []

    between = apply_price_filters(df, {"min_price": 20.0, "min_price_operator": "gte", "max_price": 30.0, "max_price_operator": "lte"})
    assert between["price"].tolist() == [22.5, 24.99, 25.0, 26.0]

    exact = apply_price_filters(df, {"exact_price": 25.0})
    assert exact["price"].tolist() == [25.0]
