# Scikit-learn – cheatsheet pro ML

> Klíčová slova: sklearn, scikit-learn, machine learning, fit, predict, transform,
> Pipeline, train_test_split, GridSearchCV, StandardScaler, RandomForest, metrics
> Verze: scikit-learn 1.3+

## Instalace a import

```python
# V .projenrc.py:
project.add_dependency("scikit-learn@^1.3")

# Import vzory:
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```

---

## Základní rozhraní – Estimator API

Všechny modely a transformátory sdílejí stejné rozhraní:

```python
model.fit(X_train, y_train)          # trénink (u transformátorů fit na train datech)
model.predict(X_test)                 # predikce (klasifikátory, regresory)
model.transform(X)                    # transformace (preprocessing)
model.fit_transform(X_train)          # fit + transform najednou
model.score(X_test, y_test)           # výchozí metrika modelu (accuracy nebo R²)
model.get_params()                    # vrátí dict parametrů
model.set_params(**params)            # nastaví parametry
```

---

## Rozdělení dat

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20 % test
    random_state=42,        # reprodukovatelnost
    stratify=y,             # zachovat poměr tříd (jen klasifikace)
)
```

---

## Preprocessing

### StandardScaler (z-score normalizace)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit na train!
X_test_scaled = scaler.transform(X_test)          # jen transform na test
```

### MinMaxScaler (rozsah 0–1)
```python
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
```

### LabelEncoder (kategorické → čísla, jen pro y)
```python
le = LabelEncoder()
y_encoded = le.fit_transform(y)        # ["cat","dog","cat"] → [0,1,0]
y_original = le.inverse_transform([0]) # [0] → ["cat"]
le.classes_                            # ["cat", "dog"]
```

### OneHotEncoder (kategorické → binary sloupce)
```python
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = ohe.fit_transform(X_categorical)
```

### ColumnTransformer (různý preprocessing pro různé sloupce)
```python
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(), ["city", "gender"]),
], remainder="passthrough")   # zbytek sloupců ponechat
```

---

## Pipeline

Pipeline řetězí preprocessing + model. **Klíčový vzor – zabraňuje data leakage.**

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test, y_test)

# Pipeline s ColumnTransformer:
full_pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(), cat_features),
    ])),
    ("classifier", LogisticRegression(max_iter=1000)),
])
```

---

## Klasifikátory

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Nejčastěji použité:
LogisticRegression(C=1.0, max_iter=1000, multi_class="auto")
RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
SVC(C=1.0, kernel="rbf", probability=True)   # probability=True pro predict_proba
KNeighborsClassifier(n_neighbors=5)
```

## Regresory

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

LinearRegression()
Ridge(alpha=1.0)          # L2 regularizace
Lasso(alpha=1.0)          # L1 regularizace (feature selection)
ElasticNet(alpha=1.0, l1_ratio=0.5)
RandomForestRegressor(n_estimators=100, random_state=42)
```

## Clustering (bez y)

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

KMeans(n_clusters=3, random_state=42, n_init="auto")
DBSCAN(eps=0.5, min_samples=5)
```

---

## Metriky

### Klasifikace
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)

accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred, average="macro")  # micro/macro/weighted
recall_score(y_true, y_pred, average="weighted")
f1_score(y_true, y_pred, average="binary")
roc_auc_score(y_true, y_prob)          # y_prob = predict_proba[:, 1]
print(classification_report(y_true, y_pred))       # přehledný výpis
confusion_matrix(y_true, y_pred)
```

### Regrese
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)          # 1.0 = perfektní, 0 = průměr, <0 = špatný
```

### Clustering
```python
from sklearn.metrics import silhouette_score, adjusted_rand_score
silhouette_score(X, labels)            # vyšší = lepší (max 1.0)
```

---

## Cross-validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Jednoduchá CV
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"CV: {scores.mean():.3f} ± {scores.std():.3f}")

# Stratified (zachová poměr tříd)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
```

---

## Hyperparameter tuning

### GridSearchCV
```python
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 5, 10],
    "preprocessor__num__with_mean": [True, False],  # parametry preprocessoru
}
grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,      # paralelně
    verbose=1,
)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
best_model = grid_search.best_estimator_
```

### RandomizedSearchCV (rychlejší pro velký param_grid)
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    "model__n_estimators": randint(50, 300),
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": randint(2, 20),
}
random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=20, cv=5, n_jobs=-1, random_state=42
)
```

---

## Uložení a načtení modelu

```python
import joblib

# Uložit
joblib.dump(pipeline, "model.joblib")

# Načíst
pipeline = joblib.load("model.joblib")
y_pred = pipeline.predict(X_test)
```

---

## Feature importance

```python
# RandomForest, GradientBoosting
importances = model.feature_importances_
feature_names = X.columns.tolist()
sorted_idx = importances.argsort()[::-1]
for i in sorted_idx[:10]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Přístup přes Pipeline
rf = pipeline.named_steps["model"]
importances = rf.feature_importances_
```
