import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('./ressources/client_data.csv')

# Sélection des caractéristiques
x = data[['income', 'product_type', 'region', 'is_client', 'family_status', 'credit_term', 'phone_operator']]
y = data['bad_client_target'] # cible

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test = train_test_split(x, test_size=0.2, random_state=42)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

# Prétraitement des caractéristiques
numeric_features = ['credit_term', 'region', 'income', 'phone_operator', 'is_client']
numeric_transformer = StandardScaler()

categorical_features = ['product_type', 'family_status']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#conversion typo data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

knn = KNeighborsClassifier(n_neighbors=5)
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', knn)])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
