# Projen Python API – Setuptools, Uv, Venv, RequirementsFile

> Zdroj: https://projen.io/docs/api/python
> Chunk 5/5 – alternativní dependency managery a packaging

## Setuptools

Packaging přes setuptools a setup.py.
Implementuje: `IPythonPackaging`
Výchozí aktivní pokud `poetry=False` a `uv=False`.

```python
project = PythonProject(
    ...,
    pip=True,
    venv=True,
    setuptools=True,     # výchozí True pokud poetry=False
    setup_config={       # dodatečná konfigurace pro setup()
        "install_requires": ["requests>=2.28"],
        "entry_points": {"console_scripts": ["my-cli=my_module.__main__:main"]},
    }
)
```

### SetuptoolsOptions

| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `generate_setup_py` | `bool` | `True` | Generovat setup.py |

---

## RequirementsFile

Spravuje requirements.txt soubor.
Automaticky vytvořen pokud `pip=True`.

### RequirementsFileOptions

| Parametr | Typ | Popis |
|----------|-----|-------|
| `filename` | `str` | Název souboru (výchozí "requirements.txt") |

---

## Uv

Moderní, rychlý správce závislostí a prostředí (náhrada pip + venv).
Implementuje: `IPythonDeps`, `IPythonEnv`, `IPythonPackaging`

```python
project = PythonProject(
    ...,
    uv=True,
    pip=False,
    venv=False,
    setuptools=False,
    uv_options={
        "generate_setup_py": False,
    }
)
```

### UvOptions

| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `generate_setup_py` | `bool` | `False` | Generovat setup.py |

### Uv metody

| Metoda | Popis |
|--------|-------|
| `add_dependency(spec)` | Přidá runtime závislost |
| `add_dev_dependency(spec)` | Přidá dev závislost |
| `install_dependencies()` | Nainstaluje závislosti |
| `setup_environment()` | Inicializuje venv přes uv |

---

## Venv

Spravuje Python virtual environment přes standardní `venv` modul.
Implementuje: `IPythonEnv`

```python
project = PythonProject(
    ...,
    venv=True,
    venv_options={"python_version": "3.13", "envdir": ".venv"}
)
```

### VenvOptions

| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `python_version` | `str` | - | Python verze pro venv |
| `envdir` | `str` | `".venv"` | Adresář virtual environment |

---

## Porovnání dependency managerů

| | pip | poetry | uv |
|--|-----|--------|-----|
| Lock file | Ne | `poetry.lock` | `uv.lock` |
| Virtual env | Venv | Interní | Interní |
| Packaging | Setuptools | Ano | Ano |
| Rychlost | Střední | Pomalý | Rychlý |
| Tento projekt | **Ne** | **Ano** | Ne |
| Neslučitelné s | - | pip, venv, setuptools | pip, venv |

---

## SetupPy

Reprezentuje `setup.py` soubor.
Automaticky generován pokud `setuptools=True`.

### SetupPyOptions

| Parametr | Typ | Popis |
|----------|-----|-------|
| `version` | `str` | Verze (výchozí z projektu) |
| `author_email` | `str` | Email autora |
| `url` | `str` | URL projektu |
| `license_` | `str` | Typ licence |
| `classifiers` | `list[str]` | PyPI classifiers |
| `name` | `str` | Název balíčku |
| `packages` | `list[str]` | Seznam packages |
| `description` | `str` | Popis |
| `package_data` | `dict` | Data soubory v balíčku |
| `scripts` | `list[str]` | CLI skripty |
| `install_requires` | `list[str]` | Závislosti |
| `homepage` | `str` | URL domovské stránky |
| `keywords` | `list[str]` | Klíčová slova |
