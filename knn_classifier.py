import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv("Iris.csv")
if "Id" in data.columns:
    data = data.drop(columns=["Id"])

X = data.drop(columns=["Species"])
y = data["Species"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nEnter flower measurements:")
sepal_length = float(input("Sepal Length: "))
sepal_width  = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width  = float(input("Petal Width: "))

user_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=X.columns
)

prediction = knn.predict(user_data)

print("Predicted Species:", le.inverse_transform(prediction))
