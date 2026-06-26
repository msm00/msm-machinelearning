# msm-machinelearning – vzory a architektura projektu

> Klíčová slova: html_to_md, scrape_to_markdown, BeautifulSoup, markdownify,
> Utils, example, projen workflow, ML pipeline pattern, projekt struktura
> Tento soubor popisuje KONKRÉTNÍ vzory tohoto projektu.

## Struktura modulu

```
msm_machinelearning/
├── __init__.py          – veřejné API modulu
├── __main__.py          – CLI vstupní bod (python -m msm_machinelearning)
├── example.py           – ukázkový kód
└── Utils/
    ├── __init__.py
    └── html_to_md.py    – HTML scraping → Markdown konverze
```

---

## Existující funkce: html_to_md.py

### `scrape_to_markdown(url, output_filename)`

Stáhne HTML z URL, odstraní navigaci/skripty, konvertuje na Markdown.

```python
from msm_machinelearning.Utils.html_to_md import scrape_to_markdown

scrape_to_markdown("https://example.com", "output.md")
```

**Závislosti:** `requests`, `beautifulsoup4`, `markdownify`

**Co dělá:**
1. GET request na URL (s browser User-Agent)
2. Odstraní `script`, `style`, `nav`, `footer`, `header`, `aside`, `form` tagy
3. Konvertuje HTML → Markdown přes `markdownify` (heading_style="ATX")
4. Vyčistí vícenásobné prázdné řádky
5. Uloží do souboru

**Vzor pro rozšíření:**

```python
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import requests

def scrape_structured(url: str) -> dict:
    """Vrátí strukturovaná data z URL."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")

    # Odstranit noise
    for el in soup(["script", "style", "nav", "footer", "aside"]):
        el.decompose()

    return {
        "title": soup.title.string if soup.title else "",
        "markdown": md(str(soup.body), heading_style="ATX"),
        "headings": [h.get_text() for h in soup.find_all(["h1", "h2", "h3"])],
    }
```

---

## Vzor pro přidání ML funkcionality

### Doporučená struktura pro ML

```
msm_machinelearning/
├── Utils/
│   ├── html_to_md.py       – scraping
│   └── text_processing.py  – čištění textu, tokenizace (nový)
├── models/                 – ML modely (nový adresář)
│   ├── __init__.py
│   ├── classifier.py       – klasifikátor dokumentů
│   └── embedder.py         – embeddingy přes Ollama
├── pipelines/              – sklearn/Ollama pipelines (nový)
│   ├── __init__.py
│   └── doc_pipeline.py     – scrape → embed → classify
├── __init__.py
└── __main__.py
```

### Vzor: Dokument klasifikátor s Ollama embeddingy

```python
# msm_machinelearning/models/embedder.py
import ollama
import numpy as np
from typing import Union

class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    def embed(self, texts: Union[str, list[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        result = ollama.embed(model=self.model, input=texts)
        return np.array(result.embeddings)
```

### Vzor: sklearn Pipeline s Ollama embeddingy

```python
# msm_machinelearning/pipelines/doc_pipeline.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
import ollama

class OllamaEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """sklearn kompatibilní transformer pro Ollama embeddingy."""

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    def fit(self, X, y=None):
        return self  # bez trénování

    def transform(self, X):
        result = ollama.embed(model=self.model, input=list(X))
        return np.array(result.embeddings)

# Použití v Pipeline:
pipeline = Pipeline([
    ("embedder", OllamaEmbeddingTransformer(model="nomic-embed-text")),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000)),
])
pipeline.fit(X_train_texts, y_train)
y_pred = pipeline.predict(X_test_texts)
```

---

## Vzor: Přidání nové závislosti do projektu

```python
# 1. Edituj .projenrc.py – přidej do deps=[]:
deps=[
    "python@>=3.13,<4.0",
    "beautifulsoup4@4.12.3",
    "markdownify@0.6.5",
    "requests@2.28.2",
    "numpy@^1.26",          # ← přidat
    "pandas@^2.1",          # ← přidat
    "scikit-learn@^1.3",    # ← přidat
    "ollama@^0.4",          # ← přidat
],

# 2. Spustit:
# npx projen
# npx projen install
```

---

## Vzor: CLI nástroj (\_\_main\_\_.py)

```python
# msm_machinelearning/__main__.py
import argparse
import sys
from msm_machinelearning.Utils.html_to_md import scrape_to_markdown

def main():
    parser = argparse.ArgumentParser(description="MSM Machine Learning nástroje")
    subparsers = parser.add_subparsers(dest="command")

    # Příkaz: scrape
    scrape_parser = subparsers.add_parser("scrape", help="Scrape URL do Markdown")
    scrape_parser.add_argument("url", help="URL ke stažení")
    scrape_parser.add_argument("--output", default="output.md", help="Výstupní soubor")

    args = parser.parse_args()

    if args.command == "scrape":
        scrape_to_markdown(args.url, args.output)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Spuštění: `python -m msm_machinelearning scrape https://example.com`

---

## Vzor: Přidání CLI task do projen

```python
# V .projenrc.py:
project.add_task(
    "scrape",
    exec="python -m msm_machinelearning scrape",
    description="Scrape URL do Markdown"
)
```

---

## Prompt vzory pro JetBrains AI s tímto projektem

### Přidat závislost
```
@file:.projenrc.py
@file:docs/llm-context/projen-cheatsheet.md

Přidej závislost scikit-learn@^1.3 do deps. Uprav jen .projenrc.py.
```

### Nová ML funkce
```
@file:msm_machinelearning/Utils/html_to_md.py
@file:docs/llm-context/project-patterns.md
@file:docs/llm-context/ollama-api.md

Přidej funkci, která vezme Markdown text, embeduje ho přes Ollama
(nomic-embed-text) a vrátí numpy array s vektorem.
```

### Sklearn pipeline
```
@file:docs/llm-context/ml-sklearn.md
@file:docs/llm-context/ollama-api.md
@file:docs/llm-context/project-patterns.md

Vytvoř sklearn Pipeline, která:
1. Embeduje texty přes OllamaEmbeddingTransformer
2. Klasifikuje je přes LogisticRegression
Použij vzory z project-patterns.md.
```

---

## Konfigurační konstanty projektu

```python
# Z .projenrc.py:
PROJECT_NAME = "msm-machinelearning"
PROJECT_VERSION = "0.1.1"
MODULE_NAME = "msm_machinelearning"
AUTHOR_NAME = "Milan Šmíd"
AUTHOR_EMAIL = "smidmi@gmail.com"
PYTHON_RANGE = ">=3.13,<4.0"
```
