# Ollama API – cheatsheet

> Klíčová slova: ollama, local LLM, REST API, python client, generate, chat,
> embeddings, qwen, stream, temperature, num_ctx, model parameters
> Zdroj: https://github.com/ollama/ollama/blob/main/docs/api.md

## Instalace Python klienta

```python
# V .projenrc.py:
project.add_dependency("ollama@^0.4")

# Import:
import ollama
from ollama import Client, AsyncClient
```

---

## REST API – přehled endpointů

| Metoda | Endpoint | Popis |
|--------|----------|-------|
| POST | `/api/generate` | Textová generace (completion) |
| POST | `/api/chat` | Chat (s historií) |
| POST | `/api/embed` | Vektorové embeddingy |
| GET | `/api/tags` | Seznam dostupných modelů |
| GET | `/api/ps` | Právě načtené modely |
| POST | `/api/show` | Info o modelu |
| POST | `/api/pull` | Stáhnout model |
| DELETE | `/api/delete` | Smazat model |
| GET | `/api/version` | Verze Ollama serveru |

Výchozí base URL: `http://localhost:11434`

---

## Python klient – základní použití

### Chat (doporučený způsob)

```python
import ollama

# Jednoduché volání (bez streamingu)
response = ollama.chat(
    model="qwen2.5:14b",
    messages=[
        {"role": "system", "content": "Jsi Python developer."},
        {"role": "user", "content": "Jak přidám závislost do projen projektu?"},
    ],
    options={"temperature": 0.3},
)
print(response.message.content)

# Se streamingem
for chunk in ollama.chat(
    model="qwen2.5:14b",
    messages=[{"role": "user", "content": "Vysvětli pandas groupby"}],
    stream=True,
):
    print(chunk.message.content, end="", flush=True)
```

### Generate (completion bez chat formátu)

```python
response = ollama.generate(
    model="qwen2.5:7b",
    prompt="def fibonacci(n):",
    system="Jsi zkušený Python developer. Piš stručný, čistý kód.",
    options={
        "temperature": 0.2,
        "num_predict": 200,    # max tokenů v odpovědi
    },
)
print(response.response)
```

### Embeddingy

```python
# Jeden text
result = ollama.embed(model="nomic-embed-text", input="Přehled strojového učení")
vector = result.embeddings[0]   # list[float]

# Batch (více textů najednou)
result = ollama.embed(
    model="nomic-embed-text",
    input=["text 1", "text 2", "text 3"],
)
vectors = result.embeddings     # list[list[float]]
```

---

## Model parametry (options)

Předávají se jako `options={}` v každém volání:

| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `temperature` | float | 0.8 | Kreativita (0 = deterministický, 2 = maximálně kreativní) |
| `top_p` | float | 0.9 | Nucleus sampling – kumul. pravděp. tokenů |
| `top_k` | int | 40 | Počet tokenů k výběru |
| `num_ctx` | int | 2048 | Délka kontextového okna (tokenů) |
| `num_predict` | int | 128 | Max tokenů v odpovědi (-1 = neomezeně) |
| `seed` | int | 0 | Seed pro reprodukovatelnost (0 = náhodný) |
| `stop` | list[str] | [] | Zastavit generování na těchto sekvencích |
| `repeat_penalty` | float | 1.1 | Penalizace opakování (1.0 = bez penalizace) |
| `repeat_last_n` | int | 64 | Okno pro repeat_penalty |
| `num_thread` | int | auto | Počet CPU vláken |
| `num_gpu` | int | 1 | Počet GPU vrstev |
| `mirostat` | int | 0 | Mirostat sampling (0/1/2) |

### Doporučená nastavení pro různé úkoly

```python
# Kód / strukturovaný výstup (deterministický)
options_code = {"temperature": 0.1, "top_p": 0.9, "seed": 42}

# Analýza / odpovědi na fakta
options_factual = {"temperature": 0.3, "top_p": 0.9}

# Kreativní psaní
options_creative = {"temperature": 0.8, "top_k": 50}

# Dlouhý kontext (pro dokumentaci)
options_long = {"temperature": 0.2, "num_ctx": 32768, "num_predict": 2048}
```

---

## Vlastní Client (jiný host/port)

```python
from ollama import Client

client = Client(host="http://localhost:11434")  # výchozí
# nebo vzdálený server:
client = Client(host="http://192.168.1.10:11434")

response = client.chat(
    model="qwen2.5:14b",
    messages=[{"role": "user", "content": "..."}],
)
```

---

## AsyncClient (pro async kód)

```python
import asyncio
from ollama import AsyncClient

async def main():
    client = AsyncClient()
    response = await client.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "Ahoj!"}],
    )
    print(response.message.content)

    # Async streaming
    async for chunk in await client.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "Vysvětli numpy broadcasting"}],
        stream=True,
    ):
        print(chunk.message.content, end="", flush=True)

asyncio.run(main())
```

---

## REST API přímé volání (requests)

```python
import requests, json

BASE = "http://localhost:11434"

# Chat
resp = requests.post(f"{BASE}/api/chat", json={
    "model": "qwen2.5:14b",
    "messages": [{"role": "user", "content": "Co je projen?"}],
    "stream": False,
    "options": {"temperature": 0.3},
})
print(resp.json()["message"]["content"])

# Embeddingy
resp = requests.post(f"{BASE}/api/embed", json={
    "model": "nomic-embed-text",
    "input": ["text 1", "text 2"],
})
vectors = resp.json()["embeddings"]

# Seznam modelů
models = requests.get(f"{BASE}/api/tags").json()["models"]
for m in models:
    print(m["name"], m["details"]["parameter_size"])
```

---

## Streaming – zpracování chunků

```python
import requests, json

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen2.5:7b", "prompt": "Napiš funkci pro výpočet faktoriálu"},
    stream=True,
)
full_text = ""
for line in resp.iter_lines():
    if line:
        chunk = json.loads(line)
        full_text += chunk.get("response", "")
        if chunk.get("done"):
            break
print(full_text)
```

---

## Správa modelů

```python
import ollama

# Seznam modelů
models = ollama.list()
for m in models.models:
    print(m.model, m.details.parameter_size, m.details.quantization_level)

# Stáhnout model
ollama.pull("qwen2.5:14b")

# Info o modelu
info = ollama.show("qwen2.5:14b")
print(info.modelfile)

# Právě načtené modely (v paměti)
running = ollama.ps()

# Smazat model
ollama.delete("qwen2.5:7b")
```

---

## Structured output (JSON formát)

```python
import json

response = ollama.chat(
    model="qwen2.5:14b",
    messages=[{
        "role": "user",
        "content": "Vrať JSON s klíči 'name' a 'score' pro 3 Python knihovny pro ML.",
    }],
    format="json",   # vynutí JSON výstup
    options={"temperature": 0.1},
)
data = json.loads(response.message.content)
```

---

## Vzor: RAG s lokálními embeddingy

```python
import ollama
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. Embed dokumenty
docs = ["Projen je konfigurace jako kód", "Numpy je knihovna pro numeriku", ...]
result = ollama.embed(model="nomic-embed-text", input=docs)
doc_vectors = np.array(result.embeddings)

# 2. Embed dotaz
query = "Jak funguje projen?"
q_result = ollama.embed(model="nomic-embed-text", input=query)
query_vector = np.array(q_result.embeddings[0])

# 3. Najít nejpodobnější dokument
similarities = [cosine_similarity(query_vector, dv) for dv in doc_vectors]
best_idx = np.argmax(similarities)
context = docs[best_idx]

# 4. Generovat odpověď s kontextem
response = ollama.chat(
    model="qwen2.5:14b",
    messages=[
        {"role": "system", "content": f"Kontext:\n{context}"},
        {"role": "user", "content": query},
    ],
    options={"temperature": 0.2},
)
print(response.message.content)
```

---

## Modely pro tento projekt (Qwen)

| Model | Velikost | num_ctx | Vhodný pro |
|-------|----------|---------|-----------|
| `qwen2.5:3b` | ~2 GB | 8 192 | Rychlé dotazy, projen cheatsheet |
| `qwen2.5:7b` | ~5 GB | 32 768 | Každodenní kód, 1–2 chunky |
| `qwen2.5:14b` | ~9 GB | 32 768 | Složitý kód, více chunků |
| `qwen2.5:32b` | ~20 GB | 32 768 | Architektura, celý projekt |
| `nomic-embed-text` | ~300 MB | - | Embeddingy pro RAG |
