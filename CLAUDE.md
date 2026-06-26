# msm-machinelearning – kontext pro AI asistenty

## Základní informace

- **Typ projektu:** Python projekt spravovaný přes [projen.io](https://projen.io)
- **Dependency manager:** Poetry (pip=False, venv=False, setuptools=False)
- **Python:** >=3.13,<4.0
- **Modul:** `msm_machinelearning`
- **Verze:** 0.1.1

## KRITICKÉ PRAVIDLO – projen

**Nikdy neupravuj tyto soubory ručně** – jsou generovány při `npx projen`:
- `pyproject.toml`
- `poetry.lock` (spravuje Poetry)
- `.projen/tasks.json`
- `.gitignore`
- `.gitattributes`

**Správný workflow:**
```
edituj .projenrc.py  →  npx projen  →  commitni změny
```

## Struktura projektu

```
.projenrc.py          ← JEDINÝ soubor pro konfiguraci projektu
msm_machinelearning/  ← Python modul
  __init__.py
  __main__.py
  example.py
  Utils/
    __init__.py
    html_to_md.py
tests/                ← pytest testy
docs/
  llm-context/        ← kompaktní dokumentace pro LLM modely
  Projen/             ← původní projen API dokumentace
pyproject.toml        ← GENEROVANÝ (neupravovat)
poetry.lock           ← GENEROVANÝ (neupravovat)
```

## Konfigurace projektu (.projenrc.py)

Projekt používá:
1. **PythonProject** s Poetry jako dependency managerem
2. **PEP 621 overrides** – metadata jsou v sekci `[project]` (ne `[tool.poetry]`)
3. **PyprojectTomlFile** komponenta pro override metadata

```python
from projen.python import PyprojectTomlFile, PythonProject

project = PythonProject(
    author_email="smidmi@gmail.com",
    author_name="Milan Šmíd",
    module_name="msm_machinelearning",
    name="msm-machinelearning",
    description="MSM test projen project",
    pip=False,
    poetry=True,
    venv=False,
    version="0.1.1",
    deps=[
        "python@>=3.13,<4.0",
        "beautifulsoup4@4.12.3",
        "markdownify@0.6.5",
        "requests@2.28.2",
    ],
)
```

## Jak přidat závislost

```python
# V .projenrc.py, přidat do deps= parametru:
deps=[
    "python@>=3.13,<4.0",
    "requests@2.28.2",
    "numpy@^1.24",   # ← přidat sem
],

# Nebo po vytvoření projektu:
project.add_dependency("numpy@^1.24")
project.add_dev_dependency("pytest-cov@^4.0")
```

Poté spustit: `npx projen`

## Spuštění tasků

```bash
npx projen          # re-synthesis (přegeneruje soubory)
npx projen test     # spustí pytest
npx projen install  # nainstaluje závislosti
npx projen build    # sestaví projekt
```

## Runtime závislosti

| Balíček | Verze | Účel |
|---------|-------|------|
| beautifulsoup4 | 4.12.3 | Parsování HTML |
| markdownify | 0.6.5 | Konverze HTML → Markdown |
| requests | 2.28.2 | HTTP requesty |

## Dev závislosti

| Balíček | Verze | Účel |
|---------|-------|------|
| projen | 0.99.58 | Project management |
| pytest | 7.4.3 | Testovací framework |

## Dokumentace pro LLM modely

V `docs/llm-context/` jsou kompaktní soubory optimalizované pro lokální modely:

### Projen dokumentace

| Soubor | Obsah | Vhodný pro |
|--------|-------|------------|
| `projen-cheatsheet.md` | Ultra-kompaktní přehled | Qwen 7B (vždy načíst) |
| `00-concepts.md` | Lifecycle, komponenty, soubory | Qwen 14B+ |
| `01-python-project.md` | PythonProject třída + options | Qwen 14B+ |
| `02-deps-poetry.md` | Poetry, Pip, Uv, Venv | Qwen 14B+ |
| `03-pyproject-toml.md` | addOverride, PEP 621 | Qwen 14B+ |
| `04-testing-tasks.md` | Pytest, Tasks, Projenrc | Qwen 14B+ |
| `05-setuptools-uv-venv.md` | Alternativní managery | Qwen 32B |
| `06-tasks-advanced.md` | Kompletní Task API | Qwen 14B+ |

### ML a Ollama dokumentace

| Soubor | Obsah | Vhodný pro |
|--------|-------|------------|
| `ml-sklearn.md` | Pipeline, fit/predict, metriky | Qwen 14B+ |
| `ml-pandas-numpy.md` | DataFrame, ndarray, vectorization | Qwen 14B+ |
| `ollama-api.md` | REST API, Python client, parametry | Qwen 7B+ |
| `project-patterns.md` | Vzory tohoto projektu, prompt šablony | Qwen 14B+ |

**Doporučení pro JetBrains AI:**
- Qwen 7B: `projen-cheatsheet.md` + `ollama-api.md` (~3 000 tokenů)
- Qwen 14B: cheatsheet + 1–2 relevantní chunky podle tématu
- Qwen 32B: cheatsheet + všechny relevantní chunky (~15 000 tokenů)

**Mapování dotaz → soubor:**
- Přidat závislost → `projen-cheatsheet.md`
- Ollama volání → `ollama-api.md`
- Sklearn pipeline → `ml-sklearn.md`
- DataFrame operace → `ml-pandas-numpy.md`
- Vzory projektu → `project-patterns.md`
