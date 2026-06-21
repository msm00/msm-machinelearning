import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re


def scrape_to_markdown(url, output_filename="vystup.md"):
    # 1. Stažení HTML
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Chyba při stahování: {response.status_code}")
        return

    # 2. Vyčištění HTML pomocí BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Odstranění běžného balastu, který v RAGu nechceš
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        element.decompose()

    # 3. Převod na Markdown
    # Uvnitř soup.encode() předáme očištěné HTML
    html_content = str(soup.body if soup.body else soup)
    markdown_text = md(html_content, heading_style="ATX")  # ATX vygeneruje klasické #, ## headings

    # 4. Drobná korekce víceúrovňových prázdných řádků
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text).strip()

    # 5. Uložení
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Uloženo do {output_filename}")


# Příklad použití:
url_k_obsahu = "https://example.com/nejaka-dokumentace"
scrape_to_markdown(url_k_obsahu, "dokumentace.md")