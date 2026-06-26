# Projen – Task API (pokročilé)

> Zdroj: https://projen.io/docs/api/projen (třída Task)
> Chunk 6 – kompletní Task API, kroky, podmínky, environment

## Task – přehled

Task je série shell příkazů a podtasků, spouštěná přes `npx projen <task>`.
Definice jsou uloženy v `.projen/tasks.json` (generovaný soubor).

```bash
npx projen         # výchozí task (synthesis)
npx projen test    # spustí task "test"
npx projen lint    # spustí vlastní task "lint"
```

---

## Vytvoření a úprava tasků

### Základní vytvoření

```python
# Jednoduchý task s jedním příkazem
task = project.add_task("lint", exec="flake8 .")

# Task s popisem
task = project.add_task(
    "format",
    exec="black . && isort .",
    description="Formátuje Python kód"
)

# Task bez počátečního příkazu (přidáme kroky postupně)
task = project.add_task("ci")
task.exec("npx projen test")
task.exec("npx projen lint")
```

### Přidání kroků k existujícímu tasku

```python
# Získat existující task
test_task = project.tasks.try_find("test")

# Přidat krok na konec
test_task.exec("pytest --cov=msm_machinelearning tests/")

# Přidat krok na začátek
test_task.prepend_exec("echo 'Spouštím testy...'")

# Přidat zprávu
test_task.say("Testy hotovy!")
```

### Reset a přepsání tasku

```python
# Smazat všechny kroky a začít znovu
test_task.reset("pytest -x tests/")

# Odebrat task úplně
project.remove_task("test")
```

---

## Task metody

### exec(command, options?)
Přidá shell příkaz jako krok.
```python
task.exec("flake8 .")
task.exec("mypy .", options={"cwd": "src/"})
```

### spawn(subtask, options?)
Spustí jiný task jako podtask.
```python
lint_task = project.add_task("lint")
test_task = project.add_task("test")

ci_task = project.add_task("ci")
ci_task.spawn(lint_task)
ci_task.spawn(test_task)
```

### say(message, options?)
Vypíše zprávu (nezastaví při chybě).
```python
task.say("Sestavuji projekt...")
task.exec("python -m build")
task.say("Hotovo!")
```

### env(name, value)
Nastaví proměnnou prostředí pro task.
```python
task.env("PYTHONPATH", "src")
task.env("LOG_LEVEL", "DEBUG")
task.env("VERSION", "$(cat VERSION)")  # shell evaluace
```

### prepend_exec / prepend_say / prepend_spawn
Přidají krok na **začátek** tasku.
```python
task.prepend_exec("echo 'Start'")
task.prepend_say("Inicializuji...")
```

### add_condition(condition)
Přidá podmínku – task se přeskočí pokud příkaz vrátí nenulový kód.
```python
# Spustit task jen pokud existuje soubor
task.add_condition("test -f pyproject.toml")

# Přeskočit pokud CI proměnná není nastavena
task.add_condition("test -n \"$CI\"")
```

---

## TaskOptions – kompletní struct

```python
project.add_task("my-task", **{
    "exec": "python script.py",      # první příkaz (zkratka)
    "description": "Popis tasku",
    "condition": "test -f .env",     # podmínka spuštění
    "cwd": "src/",                   # pracovní adresář
    "env": {                         # env proměnné
        "ENV": "production",
        "VERSION": "$(cat VERSION)", # shell evaluace
    },
    "required_env": ["API_KEY"],     # povinné env proměnné
    "receive_args": True,            # předat args z CLI do příkazu
    "args": ["--verbose"],           # pevné argumenty
    "steps": [                       # nebo přímá definice kroků
        {"exec": "flake8 ."},
        {"exec": "mypy ."},
        {"say": "Lint OK!"},
    ],
})
```

---

## TaskStep – definice kroku

Každý krok je dict s jedním z těchto typů:

```python
# exec – shell příkaz
{"exec": "python -m pytest"}

# say – zpráva
{"say": "Testy proběhly úspěšně"}

# spawn – jiný task (podle jména)
{"spawn": "test"}

# builtin – interní projen task
{"builtin": "release/resolve-version"}
```

S volitelným kontextem:
```python
{
    "exec": "pytest",
    "name": "Spustit testy",          # jméno kroku
    "cwd": "tests/",                  # pracovní adresář
    "condition": "test -n \"$CI\"",   # podmínka
    "env": {"PYTHONPATH": "src"},     # env proměnné
    "args": ["--verbose"],            # argumenty
    "receive_args": True,             # předat CLI args
}
```

---

## Práce s existujícími tasky

```python
# Najít task
task = project.tasks.try_find("test")   # vrátí Task nebo None
task = project.tasks.find("test")       # vrátí Task nebo vyhodí chybu

# Přidat krok na konkrétní pozici
task.insert_step(0, {"exec": "echo start"})  # na začátek
task.insert_step(-1, {"exec": "echo end"})   # na konec

# Odebrat krok
task.remove_step(0)      # odebere první krok

# Aktualizovat krok
task.update_step(0, {"exec": "nový příkaz"})

# Zobrazit kroky
print(task.steps)
```

---

## Vzorové tasky pro Python projekt

```python
# Linting
project.add_task("lint", exec="flake8 . && mypy .", description="Kontrola kódu")

# Formátování
project.add_task("format", exec="black . && isort .", description="Formátování kódu")

# CI task (spustí lint + test)
ci = project.add_task("ci")
ci.spawn(project.tasks.find("lint"))
ci.spawn(project.tasks.find("test"))

# Generování dokumentace
project.add_task("docs", exec="sphinx-build -b html docs/source docs/build")

# Čištění
project.add_task(
    "clean",
    exec="find . -type d -name __pycache__ -exec rm -rf {} +",
    description="Smaže __pycache__ adresáře"
)

# Task s podmínkou (jen v CI)
ci_only = project.add_task("deploy")
ci_only.add_condition("test -n \"$CI\"")
ci_only.exec("python -m twine upload dist/*")
```

---

## Vestavěné tasks Python projektu

| Task | Popis | Příkaz |
|------|-------|--------|
| `default` | Re-synthesis projektu | `npx projen` |
| `install` | Aktualizuje lockfile + instaluje | `poetry update && poetry install` |
| `install:ci` | Instaluje přesně dle lockfile | `poetry install --frozen` |
| `test` | Spustí pytest | `pytest` |
| `build` | Sestaví projekt | (compile + test + package) |
| `compile` | Kompilace | (prázdný krok pro Python) |
| `package` | Zabalí projekt | `poetry build` |
| `publish` | Publikuje na PyPI | `poetry publish` |
| `publish:test` | Publikuje na Test PyPI | `poetry publish --repository testpypi` |
