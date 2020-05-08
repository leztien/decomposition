"""
my PCA class (based on numpy.linalg.svd)
"""

import numpy as np


def make_data(m=1000, n=100, p=0.25):
    """
    Make data for PCA. Makes regression data, and thus returnes the continuous targets y
    p = percentage of redundant features
    The data (but not the target) is centered and scaled.
    """
    assert p < 0.5,"too many redundant features"
    n_redundant = int(n*p)
    n = int(n - n_redundant)
    #make X
    X = np.matmul(np.random.uniform(-1,1, size=(n,n)), np.random.randn(n,m)).T
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    #make y
    y = (X * np.random.uniform(-1,1, size=n)).sum(axis=1)
    error = np.random.randn(m) * (y.std()*0.4) + y.mean()
    y += error
    y = (y - y.min()).round(4)
    #make X-redundant
    if bool(int(n_redundant)):
        X_redundant = X[:,-n_redundant:] + np.random.normal(0, scale=np.random.uniform(0.01, 0.5, size=n_redundant), size=(m,n_redundant))
        X = np.c_[X,X_redundant]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return(X,y)

#___________________________________________________________________________

from numpy.linalg import svd
def cov(X):
    return sum(np.outer(x,x) for x in X) / (len(X)-1)


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        m,n = X.shape
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0, ddof=0)
        self._mask = self.sd==0 
        self.sd[self._mask] += 1e-10
        
        X = (X - self.mu) / self.sd
        
        E,λ,_ = svd(cov(X))

        nx = np.argsort(λ)[::-1]
        λ = λ[nx]
        E = (E.T[nx]).T
        info = self.pc_variances_ = λ / λ.sum()

        if (0 < self.n_components < 1):
            d = (np.cumsum(info) < float(self.n_components)).sum()+1
        else:
            d = min(abs(int(self.n_components)), n) or n
        self.components_ = E
        self.E = E[:,:d]
        return self

    def transform(self, X):
        X = (X - self.mu) / self.sd
        return np.matmul(self.E.T, X.T).T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def invert(self, Xpca):
        Xbac = np.matmul(self.E, Xpca.T).T
        sd = self.sd
        sd[self._mask] = 0
        return Xbac * self.sd +self.mu



def mean_squared_distance(original_data, reconstructed_data):
    """aka reconstruction error"""    
    for X in (original_data, reconstructed_data):
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0) + 1e-10
        X -= mu
        X /= sd
    reconstruction_error = (((original_data - reconstructed_data)**2).sum(axis=1)**.5).sum() / len(X)
    return float(reconstruction_error)


#==========================================================================

"""DEMO"""
import matplotlib.pyplot as plt; from mpl_toolkits.mplot3d import Axes3D

#multy-dimensional
X,y = make_data(m=1000, n=100, p=0.25)
pca = PCA(n_components=0.95)
Xpca = pca.fit_transform(X)
Xbac = pca.invert(Xpca)

print("number of principle components:", Xpca.shape[1])
print("reconstruction error:", round(mean_squared_distance(X,Xbac), 4))

#visualize cummulative variance
plt.plot(np.cumsum(pca.pc_variances_))


#VISUALIZE in 3D
plt.figure()

X,y = make_data(m=1000, n=3, p=0)

pca = PCA(n_components=2)
Xpca = pca.fit_transform(X)
Xbac = pca.invert(Xpca)

sp = plt.axes(projection='3d')
sp.plot(*X.T, '.')
sp.plot(*Xbac.T, '.', alpha=0.3)
plt.show()
