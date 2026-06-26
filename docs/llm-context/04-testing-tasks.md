# Projen Python API – Testování a Tasks

> Zdroj: https://projen.io/docs/api/python
> Chunk 4/5 – Pytest, Tasks, Projenrc

## Pytest

Automaticky přidán do projektu pokud `pytest=True` (výchozí).

```python
project = PythonProject(
    ...,
    pytest=True,
    pytest_options={
        "pytest_ini_options": {
            "testpaths": ["tests"],
            "addopts": "-v --tb=short",
            "filterwarnings": ["ignore::DeprecationWarning"],
        }
    },
)

# Přístup ke komponentě
pytest_component = project.pytest  # instance Pytest nebo None
```

### PytestOptions

| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `pytest_ini_options` | `dict` | `{}` | Obsah sekce `[tool.pytest.ini_options]` |
| `version` | `str` | - | Verze pytest (jako závislost) |

### Výsledek v pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

### Spuštění testů

```bash
npx projen test          # spustí pytest
npx projen test:watch    # watch mode (pokud nakonfigurován)
```

---

## Tasks

Projen generuje tasks (příkazy) spouštěné přes `npx projen <task>`.

### Vestavěné tasks pro Python projekt

| Task | Příkaz | Popis |
|------|--------|-------|
| `default` | `npx projen` | Spustí projen (re-synthesis) |
| `install` | `npx projen install` | Nainstaluje závislosti + aktualizuje lockfile |
| `install:ci` | `npx projen install:ci` | Instaluje dle lockfile (pro CI) |
| `build` | `npx projen build` | Sestaví projekt |
| `test` | `npx projen test` | Spustí testy |
| `compile` | `npx projen compile` | Kompilace |
| `package` | `npx projen package` | Zabalí projekt |
| `publish` | `npx projen publish` | Publikuje na PyPI (pokud poetry=True) |

### Vlastní tasks

```python
# Přidat task s jedním příkazem
project.add_task("lint", exec="flake8 .")

# Task se složitějšími kroky
lint_task = project.add_task("lint")
lint_task.exec("flake8 .")
lint_task.exec("mypy .")

# Task s popisem
project.add_task(
    "format",
    exec="black . && isort .",
    description="Formátuje kód"
)

# Task spouštějící jiný task
project.add_task("ci", exec="npx projen test && npx projen lint")

# Odebrat existující task
project.remove_task("build")
```

### TaskOptions

| Parametr | Typ | Popis |
|----------|-----|-------|
| `exec` | `str` | Shell příkaz k provedení |
| `description` | `str` | Popis tasku |
| `env` | `dict` | Environment proměnné |
| `condition` | `str` | Podmínka (shell příkaz, task se spustí jen pokud vrátí 0) |
| `cwd` | `str` | Pracovní adresář |

---

## Projenrc

Třída umožňující psát projenrc soubory v Pythonu.
Automaticky přidána pokud `projenrc_python=True` (výchozí).

```python
# Manuální konfigurace (např. jiný soubor)
from projen.python import Projenrc

Projenrc(project, options={
    "filename": ".projenrc.py",    # výchozí
    "python_exec": "python3",      # výchozí "python"
})
```

### ProjenrcOptions

| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `filename` | `str` | `".projenrc.py"` | Název sojenrc souboru |
| `python_exec` | `str` | `"python"` | Cesta k Python executable |

### Statická metoda

```python
# Zjistit instanci Projenrc pro projekt
projenrc = Projenrc.of(project)  # vrátí Projenrc nebo None
```

---

## PythonSample

Generuje ukázkový Python kód (automaticky pokud `sample=True`).

```python
# Deaktivace ukázkového kódu
project = PythonProject(..., sample=False)
```
