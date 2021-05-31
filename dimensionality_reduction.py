from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
model = PCA(n_components=2)  # Создаем экземпляр модели с гиперпараметром
model.fit(X_iris)  # Обучаем модель на данных
X_2D = model.transform(X_iris)  # Преобразуем данные в двумерные

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot(x='PCA1', y='PCA2', hue='species', data=iris, fit_reg=False)
plt.show()