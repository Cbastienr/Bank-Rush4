
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("./ressources/client_data.csv")

X = data[['credit_amount']]  # Param 'income'
y = data['bad_client_target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisatioon KNN avec le nombre de voisins (k) souhaité
k = 2  # Choisir le nombre de voisins
knn = KNeighborsClassifier(n_neighbors=k)

# training
knn.fit(X_train, y_train)

# pred sur la partie Test
y_pred = knn.predict(X_test)

#return result
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('Accuracy: {accuracy}')
print('Classification Report:\n', report)
