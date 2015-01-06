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
        self.obj = decomposition.RandomizedPCA(n_components=self.n_components, copy=self.copy, whiten=self.whiten)

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


