from sklearn.model_selection import train_test_split
from seaborn import load_dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = load_dataset('iris')
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)
model = GaussianNB()  # Создаем экземпляр модели
model.fit(X_train, y_train)  # Обучаем модель на данных
y_model = model.predict(X_test)  # Предсказываем значения для новых данных
accuracy = accuracy_score(y_test, y_model)  # Оценим точность предсказания
print(accuracy)
