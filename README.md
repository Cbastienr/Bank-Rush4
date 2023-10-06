## 🏦 Bank prediction

**Objectif :** Améliorer l'optimisation des processus de tri des "bons" et "mauvais" clients grâce à un code qui va prendre en compte des variables données.

## Features

```python
data = pd.read_csv('./ressources/client_data.csv')
```

Permet de lire le fichier CSV ou Excel (méthode à modifier en fonction du type).

```python
print("Accuracy:", accuracy)
print("Classification Report:")
```

Sort les prédictions de l'algorithme.

## Installation

Le langage [Python](https://www.python.org/downloads/) est requis, afin de lancer le programme.

Nécessité également d'installer les dépendances nécessaire :

| Dépendances  | Install                                                     |
| ------------ | ----------------------------------------------------------- |
| Pandas       | https://pandas.pydata.org/docs/getting_started/install.html |
| Scikit-learn | https://scikit-learn.org/stable/install.html                |

```sh
cd projet & pip fichier.py
```

- Chargement des données à partir du fichier CSV.

```python
data = pd.read_csv('./ressources/client_data.csv')
```

- Sélection des caractéristiques pertinentes et de la cible.

```python
x = data[['income', 'product_type', 'region', 'is_client', 'family_status', 'credit_term', 'phone_operator']]
y = data['bad_client_target']
```

- Division des données en ensembles d'entraînement et de test.

```python
x_train, x_test = train_test_split(X, test_size=0.2, random_state=42)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
```

- Prétraitement des caractéristiques numériques et catégorielles

```python
numeric_features = ['credit_term', 'region', 'income', 'phone_operator', 'is_client']
numeric_transformer = StandardScaler()

categorical_features = ['product_type', 'family_status']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

- Entraînement d'un modèle KNeighborsClassifier.

```python
knn = KNeighborsClassifier(n_neighbors=5)
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', knn)])
```

- Évaluation du modèle sur l'ensemble de test.

```python
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
```
