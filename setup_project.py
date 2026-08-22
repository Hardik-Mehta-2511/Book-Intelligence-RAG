from pathlib import Path

folders = [
    "scraper",
    "data/raw",
    "data/processed",
    "nlp",
    "rag",
    "app",
    "excel"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

Path("README.md").touch()

print("Project structure created successfully!")