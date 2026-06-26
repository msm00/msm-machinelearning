import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re


def scrape_to_markdown(url, output_filename="vystup.md"):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)

    print(f"Response status code: {response.status_code}")  # Debug informace

    if response.status_code != 200:
        print(f"Chyba při stahování: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        element.decompose()

    html_content = str(soup.body if soup.body else soup)
    markdown_text = md(html_content, heading_style="ATX")
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text).strip()

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Uloženo do {output_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scrape HTML and convert to Markdown.')
    parser.add_argument('url', type=str, help='URL of the webpage to scrape')
    parser.add_argument('--outputfilename', type=str, default="vystup.md", help='Output filename (default: vystup.md)')

    args = parser.parse_args()

    if len(sys.argv) > 1:
        url = args.url
        outputfilename = args.outputfilename if args.outputfilename else "vystup.md"
        scrape_to_markdown(url, outputfilename)