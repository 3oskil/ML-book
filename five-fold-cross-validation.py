from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=1)  # Создаем экземпляр модели

# Пятиблочная перекрестная проверка
ac_arr = cross_val_score(model, X, y, cv=5)

print(ac_arr)