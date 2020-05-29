import numpy as np
import matplotlib.pyplot as plt


def make_data():
    square = np.random.uniform(-1,1, size=(2000,2))
    distances = (square**2).sum(1)**.5
    mask = np.logical_and(distances <= 1, distances > 0.85)
    X = square[mask]
    y = np.zeros(shape=len(X))
    mask = np.logical_and(distances > 0.2, distances < 0.4)
    X = np.vstack([X, square[mask]])
    y = np.concatenate([y, np.ones(shape=sum(mask))])
    #add noise
    X = X + np.random.normal(loc=0, scale=0.03, size=X.shape)
    #rescale [-1,1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 2 - 1
    return(X,y)

#########################################################################

X,y = make_data()
(m,n),d = X.shape, 2


from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=2, kernel="rbf", gamma=10)
X_pca_sklearn = pca.fit_transform(X)

eigenvectors = alphas = pca.alphas_
lambdas = pca.lambdas_


def kernel(a,b):
    gamma = 10
    return np.exp(-gamma * ((a-b)**2).sum())


#compute the Gram mx
K = np.zeros(shape=(m,m))
for i in range(m):
    for j in range(m):
        K[i,j] = kernel(X[i], X[j])


#center the Gram max
#K = K - K.mean(axis=0)
M = np.ones((m,m)) / m
K = K - M.dot(K) - K.dot(M) + M.dot(K).dot(M)


#compute the eigenvectors
from scipy.linalg import eig, svd
λ, E = eig(K)  
nx = np.argsort(λ)[:-(d+1):-1]  # keep onla 2 dimensions/eigenvectors
λ = λ.real[nx]
E = E.real.T[nx].T

#project
X_pca_mine = np.matmul(K, E)


#show that mine eigenvectors are the same as sklearn's
M = np.abs(eigenvectors.round(4)) == np.abs(E.round(4))
print(M) # all True


#visualize
fig,axs = plt.subplots(2,2, figsize=(10,10))
for sp,nd in zip(axs.flat, (X, X_pca_sklearn, X_pca_mine)):
    sp.scatter(*nd.T, c=y, marker="o", cmap="RdBu", edgecolor='k')
    sp.axis("equal")
