from sklearn import decomposition
import numpy as np

class PCA:
    # standard Principal Component Analysis

    def __init__(self, data, n_components=None, copy=True, whiten=False):
        self.n_components=n_components
        self.copy=copy
        self.whiten=whiten

        self.data = data
        self.dataTransformed = None
        self.obj = decomposition.PCA(n_components=self.n_components, copy=self.copy, whiten=self.whiten)

    def fit_model(self):
        self.dataTransformed = self.obj.fit_transform(self.data)

    def print_results(self):
        print('\n Components')
        print(self.obj.components_)
        print('\n Explained Variance Ratio')
        print(self.obj.explained_variance_ratio_)
        print('\n')

class RPCA:
    # Randomized Principal Component Analysis

    def __init__(self, data, n_components=None, copy=True, iterated_power=3, whiten=False, random_state=None):
        self.n_components=n_components
        self.copy=copy
        self.iterated_power=iterated_power
        self.whiten=whiten
        self.random_state=random_state

        self.data = data
        self.dataTransformed = None
        self.obj = decomposition.RandomizedPCA(n_components=self.n_components, copy=self.copy, iterated_power=self.iterated_power, whiten=self.whiten, random_state=self.random_state)

    def fit_model(self):
        self.dataTransformed = self.obj.fit_transform(self.data)

    def print_results(self):
        print('\n Components')
        print(self.obj.components_)
        print('\n Explained Variance Ratio')
        print(self.obj.explained_variance_ratio_)
        print('\n')


class SPCA:
    # Sparse Principal Component Analysis

    def __init__(self, data, n_components=None, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=1, U_init=None, V_init=None, verbose=True, random_state=None):
        self.n_components=n_components
        self.alpha=alpha
        self.ridge_alpha=ridge_alpha
        self.max_iter=max_iter
        self.tol=tol
        self.method=method
        self.n_jobs=n_jobs
        self.U_init=U_init
        self.V_init=V_init
        self.verbose=verbose
        self.random_state=random_state

        self.data = data
        self.dataTransformed = None
        self.obj = decomposition.SparsePCA(n_components=self.n_components, alpha=self.alpha, ridge_alpha=self.ridge_alpha, max_iter=self.max_iter, tol=self.tol, method=self.method, n_jobs=self.n_jobs, U_init=self.U_init, V_init=self.V_init, verbose=self.verbose, random_state=self.random_state)

    def fit_model(self):
        self.dataTransformed = self.obj.fit_transform(self.data)

    def print_results(self):
        print('\n Components')
        print(self.obj.components_)
        print('\n Sum squared error')
        print(np.sum(np.power(self.obj.error_,2)))
        print('\n')

class ICA:
    # Independent Component Analysis

    def __init__(self, data, n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, w_init=None, random_state=None):
        self.n_components=n_components
        self.algorithm=algorithm
        self.whiten=whiten
        self.fun=fun
        self.fun_args=fun_args
        self.max_iter=max_iter
        self.tol=tol
        self.w_init=w_init
        self.random_state=random_state

        self.data = data
        self.dataTransformed = None
        self.obj = decomposition.FastICA(n_components=self.n_components, algorithm=self.algorithm, whiten=self.whiten, fun=self.fun, fun_args=self.fun_args, max_iter=self.max_iter, tol=self.tol, w_init=self.w_init, random_state=self.random_state)

    def fit_model(self):
        self.dataTransformed = self.obj.fit_transform(self.data)

    def print_results(self):
        print('\n Mixing matrix')
        print(self.obj.mixing_)
        print('\n')




from sklearn import manifold

class Isomap:
    # Class to interface with Isomap (Isometric Mapping) Embedding objects from scikit-learn Manifold Learning module

    def __init__(self, data, n_components=2, n_neighbors=5, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto'):
        self.n_components=n_components
        self.n_neighbors=n_neighbors 
        self.eigen_solver=eigen_solver
        self.tol_tol
        self.max_iter=max_iter
        self.path_method=path_method
        self.neighbors_algorithm=neighbors_algorithm

        self.data = data
        self.dataTransformed = None
        self.obj = manifold.Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto')

    def fit_model(self):
        self.dataTransformed = self.obj.fit_transform(self.data)

    def print_results(self): 
        print('\n Distribution Matrix')
        print(self.obj.dist_matrix_)
        print('\n Reconstruction Error')
        print(self.obj.reconstruction_error())
        print('\n')


class LocallyLinearEmbedding:
    # Class to interface with Locally Linear Embedding objects from scikit-learn Manifold Learning module

    def __init__(self, data, n_components=2, n_neighbors=5, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None):
        self.n_components=n_components
        self.n_neighbors=n_neighbors
        self.eigen_solver=eigen_solver
        self.tol=tol
        self.max_iter=max_iter
        self.method=method
        self.hessian_tol=hessian_tol
        self.modified_tol=modified_tol
        self.neighbors_algorithm=neighbors_algorithm
        self.random_state=random_state

        self.data = data
        self.dataTransformed = None
        self.obj = manifold.LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_componenets, reg=self.reg, eigen_solver=self.eigen_solver, tol=self.tol, max_iter=self.max_iter, method=self.method, hessian_tol=self.hessian_tol, modified_tol=self.modified_tol, neighbors_algorithm=self.neighbors_algorithm, random_state=self.random_state)

    def fit_model(self):
        self.dataTransformed = self.obj.fit_transform(self.data)

    def print_results(self):
        print('\n Reconstruction Error')
        print(self.obj.reconstruction_error_)
        print('\n')


class SpectralEmbedding:
    # Class to interface with Spectral Embedding objects from scikit-learn Manifold Learning module

    def __init__(self, data, n_components=2, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, n_neighbors=None):
        self.n_components=n_components
        self.affinity=affinity
        self.gamma=gamma
        self.random_state=random_state
        self.eigen_solver=eigen_solver
        self.n_neighbors=n_neighbors

        self.data = data
        self.dataTransformed = None
        self.obj = manifold.SpectralEmbedding(n_components=self.n_components, affinity=self.affinity, gamma=self.gamma, random_state=self.random_state, eigen_solver=self.eigen_solver, n_neighbors=self.n_neighbors)

    def fit_model(self):
        self.dataTransformed = self.obj.fit_transform(self.data)

    def print_results(self):
        print('\n Affinity Matrix')
        print(self.obj.affinity_matrix_)
        print('\n')


