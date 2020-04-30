"""
my PCA class (based on numpy.linalg.eig)
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

from numpy.linalg import eig

def scale(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def cov(X):
    return sum(np.outer(x,x) for x in X) / (len(X)-1)


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        m,n = X.shape
        λ,E = eig(cov(scale(X)))
        nx = np.argsort(λ)[::-1]
        λ = λ[nx]
        E = (E.T[nx]).T
        info = self.PC_variances_ = λ / λ.sum()

        if (0 < self.n_components < 1):
            d = (np.cumsum(info) < float(self.n_components)).sum()+1
        else:
            d = min(abs(int(self.n_components)), n) or n
        self.components_ = E
        self.E = E[:,:d]
        return self

    def transform(self, X):
        return np.matmul(self.E.T, X.T).T

    def fit_transform(self, X):
        return np.matmul(self.fit(X).E.T, X.T).T

    def invert(self, Xpca):
        return np.matmul(self.E, Xpca.T).T

#==========================================================================

def main():
    X,y = make_data(m=1000, n=100, p=0.25)

    pca = PCA(n_components=0.99)
    Xpca = pca.fit_transform(X)
    Xbac = pca.invert(Xpca)


    #VISUALIZE
    import matplotlib.pyplot as plt

    X,y = make_data(m=1000, n=3, p=0)

    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(X)
    Xbac = pca.invert(Xpca)

    from mpl_toolkits.mplot3d import Axes3D
    sp = plt.axes(projection='3d')
    sp.plot(*X.T, '.')
    sp.plot(*Xbac.T, '.', alpha=0.3)
    plt.show()


if __name__=="__main__":main()
