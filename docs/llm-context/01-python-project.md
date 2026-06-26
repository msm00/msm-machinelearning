# Projen Python API – PythonProject

> Zdroj: https://projen.io/docs/api/python
> Chunk 1/5 – třída PythonProject a její options

## PythonProject

Hlavní třída pro Python projekty v projen.

```python
from projen.python import PythonProject

project = PythonProject(options)
project.synth()
```

### Metody

| Metoda | Signatura | Popis |
|--------|-----------|-------|
| `add_dependency` | `(spec: str)` | Přidá runtime závislost ve formátu `modul@semver` |
| `add_dev_dependency` | `(spec: str)` | Přidá dev závislost |
| `add_task` | `(name: str, props?) -> Task` | Přidá nový task |
| `remove_task` | `(name: str) -> Task` | Odebere task |
| `add_git_ignore` | `(pattern: str)` | Přidá vzor do .gitignore |
| `synth` | `()` | Spustí syntézu – vygeneruje všechny soubory |
| `try_find_file` | `(file_path: str) -> FileBase` | Najde soubor v projektu |
| `try_remove_file` | `(file_path: str) -> FileBase` | Najde a odebere soubor |

### Properties

| Property | Typ | Popis |
|----------|-----|-------|
| `module_name` | `str` | Python název modulu (hyphens → underscores) |
| `version` | `str` | Verze balíčku (semver) |
| `deps_manager` | `IPythonDeps` | API pro správu závislostí |
| `env_manager` | `IPythonEnv` | API pro Python runtime prostředí |
| `packaging_manager` | `IPythonPackaging?` | API pro packaging (jen LIB projekty) |
| `pytest` | `Pytest?` | Pytest komponenta |
| `sample_testdir` | `str` | Adresář s testy (výchozí: "tests") |
| `components` | `list[Component]` | Všechny komponenty projektu |
| `tasks` | `Tasks` | Správce tasků |
| `gitignore` | `IgnoreFile` | .gitignore soubor |

---

## PythonProjectOptions – úplný seznam

### Povinné parametry
| Parametr | Typ | Popis |
|----------|-----|-------|
| `name` | `str` | Název projektu |
| `module_name` | `str` | Python modul (jen alfanumerické znaky a podtržítka) |
| `author_name` | `str` | Jméno autora |
| `author_email` | `str` | Email autora |
| `version` | `str` | Verze (semver, výchozí "0.1.0") |

### Závislosti
| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `deps` | `list[str]` | `[]` | Runtime závislosti (`modul@semver`) |
| `dev_deps` | `list[str]` | `[]` | Dev závislosti |

### Dependency manager (vzájemně výlučné!)
| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `pip` | `bool` | `True`* | Pip + requirements.txt (*False pokud poetry=True) |
| `poetry` | `bool` | `False` | Poetry (neslučitelné s pip, venv, setuptools) |
| `uv` | `bool` | `False` | uv (neslučitelné s pip, venv) |
| `venv` | `bool` | `True`* | venv pro virtual environment (*False pokud poetry=True) |
| `setuptools` | `bool` | `True`* | setuptools/setup.py (*False pokud poetry=True) |

### Testování
| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `pytest` | `bool` | `True` | Zapnout pytest |
| `pytest_options` | `PytestOptions` | - | Konfigurace pytest (viz 04-testing.md) |
| `sample` | `bool` | `True` | Generovat ukázkový kód a testy |
| `sample_testdir` | `str` | `"tests"` | Adresář pro testy |

### Metadata balíčku
| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `description` | `str` | - | Krátký popis balíčku |
| `license` | `str` | - | SPDX identifikátor (např. "MIT", "Apache-2.0") |
| `homepage` | `str` | - | URL webu projektu |
| `classifiers` | `list[str]` | - | PyPI trove classifiers |
| `package_name` | `str` | - | Název PyPI balíčku (pokud se liší od name) |

### Projenrc
| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `projenrc_python` | `bool` | `True` | Použít .projenrc.py (Python) |
| `projenrc_python_options` | `ProjenrcOptions` | - | Konfigurace projenrc |
| `projenrc_ts` | `bool` | `False` | Použít .projenrc.ts (TypeScript) |
| `projenrc_js` | `bool` | `False` | Použít .projenrc.js (JavaScript) |

### GitHub integrace
| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `github` | `bool` | `True` | GitHub integrace (CI workflows atd.) |
| `github_options` | `GitHubOptions` | - | Konfigurace GitHub integrace |
| `vscode` | `bool` | `True` | VS Code integrace |
| `dev_container` | `bool` | `False` | Devcontainer (GitHub Codespaces) |

### Ostatní
| Parametr | Typ | Výchozí | Popis |
|----------|-----|---------|-------|
| `outdir` | `str` | `"."` | Výstupní adresář projektu |
| `python_exec` | `str` | `"python"` | Cesta k Python executable |
| `poetry_options` | `PoetryPyprojectOptionsWithoutDeps` | - | Další Poetry options (viz 02-deps-poetry.md) |
| `uv_options` | `UvOptions` | - | Další uv options |
| `commit_generated` | `bool` | `True` | Commitovat generované soubory do gitu |
| `renovatebot` | `bool` | `False` | Renovatebot pro dependency upgrades |
