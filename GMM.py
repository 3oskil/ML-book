# Кластеризация. Gaussian mixture model, GMM
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from dimensionality_reduction import X_2D

iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']

model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_iris)
y_gmm = model.predict(X_iris)

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
iris['cluster'] = y_gmm
sns.lmplot(x='PCA1', y='PCA2', data=iris, hue='species',
           col='cluster', fit_reg=False)
plt.show()
