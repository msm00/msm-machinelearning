# Projen Python API – Dependency Management

> Zdroj: https://projen.io/docs/api/python
> Chunk 2/5 – Poetry, Pip, Uv, Venv třídy a jejich options

## Poetry

Spravuje závislosti, virtual environment a packaging přes Poetry CLI.
Implementuje: `IPythonDeps`, `IPythonEnv`, `IPythonPackaging`

```python
# Aktivace přes PythonProject options (doporučený způsob):
project = PythonProject(
    ...,
    poetry=True,
    pip=False,
    venv=False,
    setuptools=False,   # Poetry je neslučitelný s pip/venv/setuptools
)
```

### Metody Poetry

| Metoda | Popis |
|--------|-------|
| `add_dependency(spec: str)` | Přidá runtime závislost (`modul@semver`) |
| `add_dev_dependency(spec: str)` | Přidá dev závislost |
| `install_dependencies()` | Nainstaluje závislosti (voláno při post-synthesis) |
| `setup_environment()` | Inicializuje venv (pokud neexistuje) |

### Properties Poetry

| Property | Typ | Popis |
|----------|-----|-------|
| `install_ci_task` | `Task` | Instaluje dle lockfile (pro CI) |
| `install_task` | `Task` | Aktualizuje lockfile a instaluje |
| `publish_task` | `Task` | Publikuje na PyPI |
| `publish_test_task` | `Task` | Publikuje na Test PyPI |

### PoetryOptions / PoetryPyprojectOptionsWithoutDeps

Předávají se přes `poetry_options` v `PythonProjectOptions`:

```python
project = PythonProject(
    ...,
    poetry=True,
    poetry_options={
        "authors": ["Jméno <email@example.com>"],
        "license": "MIT",
        "homepage": "https://example.com",
        "repository": "https://github.com/user/repo",
        "documentation": "https://docs.example.com",
        "keywords": ["ml", "python"],
        "classifiers": ["Development Status :: 3 - Alpha"],
        "packages": [{"include": "my_module"}],
        "include": ["LICENSE"],
        "exclude": ["tests/"],
        "source": [{"name": "private", "url": "https://..."}],
        "scripts": {"my-cli": "my_module.__main__:main"},
        "extras": {"all": ["optional-dep"]},
        "plugins": {},
    }
)
```

---

## Pip

Spravuje závislosti přes requirements.txt a pip CLI.
Implementuje: `IPythonDeps`

```python
# Aktivace (výchozí pokud poetry=False):
project = PythonProject(..., pip=True, venv=True)
```

### Metody Pip

| Metoda | Popis |
|--------|-------|
| `add_dependency(spec: str)` | Přidá do requirements.txt |
| `add_dev_dependency(spec: str)` | Přidá dev závislost |
| `install_dependencies()` | Spustí `pip install -r requirements.txt` |

### PipOptions

| Parametr | Typ | Popis |
|----------|-----|-------|
| *(žádné specifické options)* | - | Pip nemá konfigurovatelné options |

---

## Uv

Moderní rychlý nástroj pro správu závislostí, venv a packaging.
Implementuje: `IPythonDeps`, `IPythonEnv`, `IPythonPackaging`

```python
project = PythonProject(
    ...,
    uv=True,
    pip=False,
    venv=False,
)
```

### UvOptions

| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `generate_setup_py` | `bool` | `False` | Generovat setup.py |

---

## Venv

Spravuje Python virtual environment přes venv modul.
Implementuje: `IPythonEnv`

```python
# Aktivace (výchozí s pip):
project = PythonProject(..., venv=True, venv_options={"python_version": "3.13"})
```

### VenvOptions

| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `python_version` | `str` | - | Python verze pro venv (např. "3.13") |
| `envdir` | `str` | `".venv"` | Adresář pro virtual environment |

---

## IPythonDeps interface

Rozhraní implementované Poetry, Pip i Uv:

```python
deps_manager = project.deps_manager   # vrátí instanci Poetry/Pip/Uv

deps_manager.add_dependency("requests@^2.28")
deps_manager.add_dev_dependency("pytest@^7.4")
deps_manager.install_dependencies()
```

---

## Formát závislostí (semver)

```
"requests@^2.28"       # >= 2.28, < 3.0
"numpy@~1.24.0"        # >= 1.24.0, < 1.25.0
"pandas@>=1.5,<2.0"    # rozsah
"scikit-learn@1.2.2"   # přesná verze
"python@>=3.13,<4.0"   # Python samotný (pro Poetry)
```
