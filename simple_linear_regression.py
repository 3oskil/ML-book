import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Генерируем выборку
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)

# Создаем экземпляр класса LinearRegression,
# с помощью гиперпараметра fit_intercept выполняем подбор
# точки пересечения с осью координат
model = LinearRegression(fit_intercept=True)

# Изменим размерность матрицы признаков на нужную нам
X = x[:, np.newaxis]

# Обучение модели
model.fit(X, y)

# Угловой коэффициент
print(model.coef_)

# Точка пересечения с осью ординат
print(model.intercept_)

# Предскажем данные на основе построенной модели
x_fit = np.linspace(-1, 11)
X_fit = x_fit[:, np.newaxis]
y_fit = model.predict(X_fit)

plt.scatter(x, y)
plt.plot(x_fit, y_fit)
plt.show()