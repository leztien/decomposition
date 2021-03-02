

from random import gauss, uniform, randint
from functools import singledispatch
from math import sin, cos, radians, sqrt
from functools import reduce, wraps


def transpose(mx):
    m,n = len(mx), len(mx[0])
    return [[mx[i][j] for i in range(m)] for j in range(n)]

def vector_addition(u,v):
    assert len(u)==len(v),"vectors must be of equal lengths"
    return [u+v for u,v in zip(u,v)]

def vector_subtraction(u,v):
    assert len(u)==len(v),"vectors must be of equal lengths"
    return [u-v for u,v in zip(u,v)]

def scalar_multiplication(c, v):
    assert isinstance(c, (int, float)) and isinstance(v, (list, tuple)), "bad input"
    return [c*v for v in v]


def dot(u,v):
    assert len(u)==len(v),"vectors must be of the same length"
    return sum(u*v for u,v in zip(u,v))

def matmul(A,B):
    assert len(A[0]) == len(B), "incompatible matreces"
    m,n = len(A), len(B[0])
    B = transpose(B)
    return [[dot(A[i], B[j]) for j in range(n)] for i in range(m)]


@singledispatch
def make_rotation_matrix(arg):
    raise TypeError("bad input")

@make_rotation_matrix.register(tuple)
@make_rotation_matrix.register(list)
def _(rotation_plane:tuple, n:int, angle:float):
    i,j = rotation_plane
    r = radians(angle)
    assert i!=j and 0<=i<n and 0<=j<n, "bad input"
    permutations = sorted((a,b) for b in (i,j) for a in (i,j))
    funcs = (cos, lambda r: -sin(r), sin, cos)
    mx = [[1 if i==j else 0 for j in range(n)] for i in range(n)]   # id matrix
    for (i,j),func in zip(permutations, funcs):
        mx[i][j] = func(r)
    return mx

@make_rotation_matrix.register(int)
def _(n:int):
    rotation_planes = sum([[(i,j) for j in range(i+1,n)] for i in range(n-1)], [])
    angles = [uniform(-180, 180) for _ in range(len(rotation_planes))]
    RR = [make_rotation_matrix(rotation_plane, n, angle) for (rotation_plane, angle) in zip(rotation_planes, angles)]
    return reduce(matmul, RR)


def make_data(m, n):
    sigmas = [uniform(0, n) for _ in range(n)]
    X = [[gauss(0, sigma) for i in range(m)] for j,sigma in zip(range(n), sigmas)]
    R = make_rotation_matrix(n)
    X = matmul(R, X)
    mins = [min(feature) for feature in X]  # Shift the data center
    return [vector_subtraction(x, mins) for x in transpose(X)]


def center_data(data):
    means = [sum(feature)/len(feature) for feature in transpose(data)]
    return [vector_subtraction(x, means) for x in data]

def vector_length(v):
    return sqrt(sum(v**2 for v in v))

def normalize_vector(v):
    m = vector_length(v)
    return [v/m for v in v]

def vector_variance(w, data):   # data must be centered
    w = normalize_vector(w)
    return sum(dot(w,x)**2 for x in data) / (len(data) - 1)


def vector_variance_gradient(w, data):  # data must be centered
    w = normalize_vector(w)
    return scalar_multiplication(1/len(data), reduce(vector_addition, (scalar_multiplication(dot(x,w), x) for x in data)))


def gradient_step(w, g, eta=0.1):
    return vector_addition(w, scalar_multiplication(eta, g))


def best_principal_component(data, n_iter=1000):
    w = normalize_vector(data[randint(0, len(data))])   # a random vector from the data
    for epoch in range(n_iter):
        v = vector_variance(w, data)   # variance
        g = vector_variance_gradient(w, data)
        w = gradient_step(w, g)
        if(epoch % 100 == 0): print(epoch, round(v,2), g, round(vector_length(g),2))
    return (normalize_vector(w), v)   # returns  eigenvector and variance along this vector


def project_vector(v, w):
    """Project vector v upon a given vector w"""
    projection_length = dot(v,w)   # w should be already normalized!
    return scalar_multiplication(projection_length, w)
    
def collapse_vectors(data, w):
    """collapse vectors upon the orthogonal compliment of w"""
    return [vector_subtraction(x, project_vector(x, w)) for x in data]


def pca(data):
    n = len(data[0])
    components = []
    variances = []
    for _ in range(n):
        component, variance = best_principal_component(data)
        components.append(component)   # component is already normalized
        variances.append(variance)
        data = collapse_vectors(data, component)
    if sorted(variances, reverse=True) != variances:
        from warnings import warn
        warn("variances not ordered", Warning)
    return components  # eigenvectors (arranged horizontally)


def transform_vector(vector, components):
    """Projects the vector upon the eigenvectors (i.e. components)"""
    return [dot(vector, component) for component in components]


def transform_data(data, components, n_dimensions=None):
    """Projects vectors (i.e. data) upon the eigenvectors (i.e. components)"""
    n_dimensions = n_dimensions or len(components)
    return [transform_vector(vector, components[:n_dimensions]) for vector in data]



m,n = 400, 3
X = make_data(m, n)
X = center_data(X)
E = pca(data=X)   # components


# Demo
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
xx,yy,zz = transpose(X)
ax = plt.axes(projection='3d')

mn, *_, mx = sorted(sum(X, []))
ax.set_xlim(mn,mx); ax.set_ylim(mn,mx); ax.set_zlim(mn,mx)
ax.scatter(xx,yy,zz, marker='.')
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")



ax.plot(*[0,0,0], 'or')
for e,m in zip(E, [5,3,2]):
    e = scalar_multiplication(m, normalize_vector(e))
    ax.plot(*zip([0,0,0], e), 'red')

plt.show()

# Second plot
plt.figure()
XX = transform_data(X, E, 2)
plt.scatter(*transpose(XX))
plt.axis('equal')


