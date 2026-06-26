# Pandas + NumPy – cheatsheet pro ML

> Klíčová slova: pandas, numpy, DataFrame, Series, array, ndarray, read_csv,
> groupby, merge, loc, iloc, fillna, dropna, reshape, vectorization
> Verze: pandas 2.x, numpy 1.24+

## Import

```python
import pandas as pd
import numpy as np
```

---

## PANDAS – DataFrame

### Vytvoření a načtení

```python
# Ze slovníku
df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

# Z CSV / JSON / Excel
df = pd.read_csv("data.csv", sep=",", encoding="utf-8", index_col=0)
df = pd.read_json("data.json")
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Uložení
df.to_csv("out.csv", index=False)
df.to_json("out.json", orient="records", force_ascii=False)

# Základní info
df.shape          # (řádky, sloupce)
df.dtypes         # typy sloupců
df.info()         # přehled + paměť
df.describe()     # statistiky numerických sloupců
df.head(5)        # prvních 5 řádků
df.columns.tolist()
```

### Výběr dat

```python
# Sloupce
df["col"]                    # Series
df[["col1", "col2"]]         # DataFrame

# Řádky – loc (label), iloc (index)
df.loc[0]                    # řádek podle labelu
df.loc[0:5, "col"]           # řádky 0–5, sloupec col
df.iloc[0]                   # první řádek
df.iloc[0:5, 0:3]            # řádky 0–4, sloupce 0–2
df.iloc[-1]                  # poslední řádek

# Boolean indexing (filtrování)
df[df["age"] > 30]
df[(df["age"] > 30) & (df["city"] == "Praha")]
df[df["name"].isin(["Alice", "Bob"])]
df[df["val"].between(10, 50)]

# Dotaz (čitelná alternativa)
df.query("age > 30 and city == 'Praha'")
```

### Úprava dat

```python
# Přidat / přepsat sloupec
df["new_col"] = df["a"] + df["b"]
df["ratio"] = df["a"] / df["b"].replace(0, np.nan)

# Přejmenovat
df.rename(columns={"old": "new"}, inplace=True)

# Smazat
df.drop(columns=["col1", "col2"], inplace=True)
df.drop(index=[0, 5], inplace=True)

# Změna typu
df["col"] = df["col"].astype(int)
df["date"] = pd.to_datetime(df["date"])
df["cat"] = df["cat"].astype("category")

# Apply – vlastní funkce
df["upper"] = df["name"].apply(str.upper)
df["score"] = df.apply(lambda row: row["a"] * row["b"], axis=1)

# Map (jen Series)
df["mapped"] = df["category"].map({"A": 1, "B": 2, "C": 3})
```

### Chybějící hodnoty

```python
df.isnull().sum()                        # počet NaN na sloupec
df.isnull().sum().sum()                  # celkem NaN
df.dropna()                              # smazat řádky s NaN
df.dropna(subset=["col1", "col2"])       # smazat jen kde NaN v těchto sloupcích
df.fillna(0)                             # nahradit 0
df.fillna(df.mean(numeric_only=True))    # nahradit průměrem
df["col"].fillna(method="ffill")         # forward fill (deprecated → ffill())
df["col"].ffill()                        # pandas 2.x způsob
df["col"].bfill()                        # backward fill
```

### Groupby a agregace

```python
# Základní agregace
df.groupby("city")["sales"].sum()
df.groupby("city")["sales"].agg(["mean", "std", "count"])

# Více sloupců
df.groupby(["city", "year"]).agg(
    total_sales=("sales", "sum"),
    avg_price=("price", "mean"),
    count=("id", "count"),
).reset_index()

# Transform – zachová original index (pro přidání sloupce)
df["city_avg"] = df.groupby("city")["sales"].transform("mean")
```

### Merge / Join

```python
# Merge (jako SQL JOIN)
result = pd.merge(df1, df2, on="id", how="inner")  # inner/left/right/outer
result = pd.merge(df1, df2, left_on="id", right_on="user_id")

# Concat (spojit řádky nebo sloupce)
combined = pd.concat([df1, df2], axis=0, ignore_index=True)  # řádky
combined = pd.concat([df1, df2], axis=1)                      # sloupce
```

### Pivot a reshape

```python
# Pivot table
pivot = df.pivot_table(
    values="sales", index="city", columns="year", aggfunc="sum", fill_value=0
)

# Melt (wide → long format)
long_df = df.melt(id_vars=["id"], value_vars=["2022", "2023"],
                  var_name="year", value_name="sales")
```

### Příprava pro sklearn

```python
# Oddělení features a targetu
X = df.drop(columns=["target"])
y = df["target"]

# Numerické a kategorické sloupce
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Převod na numpy array
X_array = X.values                  # nebo X.to_numpy()
y_array = y.values
```

---

## NUMPY – ndarray

### Vytvoření

```python
a = np.array([1, 2, 3])                    # 1D
b = np.array([[1, 2], [3, 4]])             # 2D (matice)
np.zeros((3, 4))                           # samé nuly
np.ones((3, 4))                            # samé jedničky
np.full((3, 4), 7)                         # samé sedmičky
np.eye(3)                                  # jednotková matice
np.arange(0, 10, 2)                        # [0, 2, 4, 6, 8]
np.linspace(0, 1, 50)                      # 50 rovnoměrně rozdělených bodů
np.random.seed(42)
np.random.rand(3, 4)                       # uniformní 0–1
np.random.randn(3, 4)                      # normální rozdělení
np.random.randint(0, 10, size=(3, 4))
```

### Tvar a reshape

```python
a.shape                     # (řádky, sloupce)
a.ndim                      # počet dimenzí
a.dtype                     # datový typ
a.reshape(4, 3)             # změna tvaru (zachová data)
a.flatten()                 # vše do 1D
a.T                         # transpozice
a[:, np.newaxis]            # přidat dimenzi (2D sloupcový vektor)
a.squeeze()                 # odstraní dimenze velikosti 1
```

### Indexing a slicing

```python
a[0]                        # první prvek
a[-1]                       # poslední prvek
a[1:4]                      # řez (jako Python list)
b[0, :]                     # první řádek
b[:, 1]                     # druhý sloupec
b[1:3, 0:2]                 # podmatice
a[a > 5]                    # boolean indexing
a[[0, 2, 4]]                # fancy indexing
```

### Matematické operace

```python
# Prvkové operace (broadcasting)
a + b                       # sčítání
a * b                       # násobení (prvkové!)
a ** 2                      # druhá mocnina
np.sqrt(a)                  # odmocnina
np.log(a)                   # přirozený logaritmus
np.exp(a)                   # e^x
np.abs(a)                   # absolutní hodnota

# Lineární algebra
np.dot(a, b)                # skalární součin / maticové násobení
a @ b                       # maticové násobení (Python 3.5+)
np.linalg.inv(b)            # inverzní matice
np.linalg.det(b)            # determinant
np.linalg.norm(a)           # norma vektoru

# Statistika
np.sum(a)                   # součet
np.mean(a)                  # průměr
np.std(a)                   # směrodatná odchylka
np.var(a)                   # rozptyl
np.min(a), np.max(a)        # min, max
np.argmin(a), np.argmax(a)  # index min, max
np.median(a)                # medián
np.percentile(a, 75)        # 75. percentil
np.unique(a)                # unikátní hodnoty
np.bincount(a)              # počet výskytů (pro celá čísla)

# Po ose (axis=0 = po sloupcích, axis=1 = po řádcích)
np.sum(b, axis=0)           # součet každého sloupce
np.mean(b, axis=1)          # průměr každého řádku
```

### Spojování a rozdělování

```python
np.concatenate([a, b], axis=0)       # spojit podél osy
np.vstack([a, b])                    # vertikálně (řádky)
np.hstack([a, b])                    # horizontálně (sloupce)
np.stack([a, b], axis=0)             # nová dimenze
np.split(a, 3)                       # rozdělit na 3 části
```

### Broadcasting pravidlo

```python
# NumPy automaticky "rozšíří" pole menší dimenze
a = np.array([[1], [2], [3]])   # shape (3, 1)
b = np.array([10, 20, 30])      # shape (3,) → (1, 3)
a + b   # výsledek shape (3, 3) – každá kombinace

# Normalizace řádků (broadcasting):
X = X - X.mean(axis=0)         # odečíst průměr každého sloupce
X = X / X.std(axis=0)          # podělit std každého sloupce
```

### Výkon – vektorizace místo smyček

```python
# POMALÉ (Python smyčka):
result = [x**2 for x in data]

# RYCHLÉ (numpy vektorizace):
result = np.array(data) ** 2

# np.vectorize pro vlastní funkci (pohodlné, ne nejrychlejší):
vfunc = np.vectorize(lambda x: x**2 if x > 0 else 0)
result = vfunc(data)

# np.where (podmíněné přiřazení):
result = np.where(a > 0, a, 0)  # kladná čísla zachovat, záporná → 0
```
