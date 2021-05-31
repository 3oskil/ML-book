from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)  # Разделяем данные: по 50% в каждом из наборов

model = KNeighborsClassifier(n_neighbors=1)  # Создаем экземпляр модели

model.fit(X1, y1)  # Обучаем модель на одном из наборов данных

y2_model = model.predict(X2)  # Оцениваем работу модели на другом наборе
print(accuracy_score(y2, y2_model))

