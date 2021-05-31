from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=1)  # Создаем экземпляр модели

# Перекрестная проверка по всем без одного
scores = cross_val_score(model, X, y, cv=LeaveOneOut())

print(scores, scores.mean(), sep='\n')