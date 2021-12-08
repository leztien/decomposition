


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=4)



class LDA:
    def __init__(self, k=None):
        self.k = k
        self._scaler = None
        
    def fit(self, X, y):
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)
        
        # Calculate the class means
        mus = []
        for c in range(len(set(y))):
            mu = X[y == c].mean(axis=0)
            mus.append(mu)
        
        # Calculate covariance matreces
        C = [(1/len(X[y==c])) + sum((x-m)[:,None] @ (x-m)[None,:] for x in X[y==c]) for (c,m) in enumerate(mus)]
        
        # Compute the Within scatter matrix
        W = sum(C)
        
        # Compute the global mean
        mu = np.mean(X, axis=0)
        
        # Compute the between scatter matrix
        B = sum(len(X[y==c]) * ((m-mu)[:,None] @ (m-mu)[None,:]) for c,m in enumerate(mus))
        
        # Multiply the two scatter matreces
        S = np.linalg.inv(W) @ B
        
        # Compute egens
        l,E = np.linalg.eig(S)
        l = [l.real for l in l]
        nx = np.argsort(l)[::-1]
        
        k = self.k or (len(set(y)) - 1)
        nx = nx[:k]
        self.E = E.real[:,nx]
        return self
        
    def transform(self, X):
        X = self._scaler.transform(X)
        return X @ self.E
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)


###############################################################



df = pd.read_csv('~/Datasets/wine.data', header=None)
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values - 1
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, stratify=y)



lda = LDA()
Xlda = lda.fit_transform(Xtrain, ytrain)
X_test_lda = lda.transform(Xtest)

# Plot
markers = ('o', '+', 'v')
colors = "rgb"

for c in range(len(set(ytrain))):
    plt.plot(*Xlda[ytrain==c].T, 
             marker=markers[c], 
             color=colors[c], 
             linewidth=0,
             label=c)
plt.title("Train data")
plt.legend()


# Test data
plt.figure()
for c in range(len(set(ytrain))):
    plt.plot(*X_test_lda[ytest==c].T, 
             marker=markers[c], 
             color=colors[c], 
             linewidth=0,
             label=c)
plt.title("Test data")
plt.legend()


########################################################################

def load_from_github(url):
    from urllib.request import urlopen
    from os import remove
    
    obj = urlopen(url)
    assert obj.getcode()==200,"unable to open"

    s = str(obj.read(), encoding="utf-8")
    NAME = "_temp.py"
    with open(NAME, mode='wt', encoding='utf-8') as fh: fh.write(s)
    module = __import__(NAME[:-3])
    remove(NAME)
    return module


url = "https://raw.githubusercontent.com/leztien/synthetic_datasets/master/make_data_for_classification.py"
module = load_from_github(url)
X,y = module.make_data_for_classification(m=400, n=10, k=4, blobs_density=0.1)

lda = LDA(k=2)
Xlda = lda.fit_transform(X, y)

#plot
plt.figure()
markers = ('o', '+', 'v', 's')
colors = "rgby"
for c in range(len(set(y))):
    plt.plot(*Xlda[y==c].T, 
             marker=markers[c], 
             color=colors[c], 
             linewidth=0,
             label=c)
plt.title("Train data")
plt.legend()
