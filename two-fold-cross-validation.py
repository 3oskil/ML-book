from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)  # Разделяем данные: по 50% в каждом из наборов

model = KNeighborsClassifier(n_neighbors=1)  # Создаем экземпляр модели

# Выполняем две попытки проверки, попеременно используя каждую половину данных в качестве отложенного набора данных
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)

print(accuracy_score(y1, y1_model), accuracy_score(y2, y2_model))