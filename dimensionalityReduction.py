import numpy as np
import visualizations as viz

# Dimensionality reduction classes using sklearn decomposition package
from sklearn import decomposition

class dimenReductClass:
    """Base class for cluster analysis.
    The init method must always be overridden by the derived class, since each clustering 
    technique requires its own scikit-learn object with different default input variables.
    """

    def init(self): # Should always be ovverridden by derived classes
        # May contain numerous variables for a given analysis method, but will always include:
        self.data ##!< Raw 2-dimensional data matrix from dataset.data
        self.dataTransformed ##<! Primary output of clustering analysis: the index of the closest cluster
        self.obj ##<! The classification object for a given method from scikit-learn
                                                                        
    def fit_model(self):
        """Run the cluster analysis method, and return the list of closest cluster indices.

        All dimensionality reduction objects from scikit-learn have consistent fit_transform 
        methods that accept the 2-d data matrix as input and return the transformed data matrix
        (e.g. the principal components)
        """
        self.dataTransformed = self.obj.fit_transform(self.data)

    def print_results(self): # May need to be overridden by derived classes
        """Print the list of closest cluster indices, which is the common output of all clustering methods"""
        print '\n Components' 
        print(self.obj.components_)
        print '\n Explained Variance Ratio'
        print(self.obj.explained_variance_ratio_)
        print '\n'

    def plot_results(self):
        """ Plot the first two extracted components against each other

        For techniques such as PCA that return components by weight, these first two components
        should capture most of the variance in the dataset. However, this may not be the case for
        techniques that return unsorted components, such as ICA.

        Uses crossplot_components class from visualizations.py
        """
        viz.crossplot_components(self.dataTransformed[:,0],self.dataTransformed[:,1]).plot()
        pass

    

class PCA(dimenReductClass):
    # standard Principal Component Analysis

    def __init__(self, data, n_components=None, copy=True, whiten=False):
        self.n_components=n_components
        self.copy=copy
        self.whiten=whiten

        self.data = data
        self.dataTransformed = None
        self.obj = decomposition.PCA(n_components=self.n_components, copy=self.copy, whiten=self.whiten)


class RPCA(dimenReductClass):
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

class SPCA(dimenReductClass):
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

    def print_results(self):
        print '\n Components'
        print(self.obj.components_)
        print '\n Sum squared error'
        print(np.sum(np.power(self.obj.error_,2)))
        print '\n'


class ICA(dimenReductClass):
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

    def print_results(self):
        print '\n Mixing matrix'
        print(self.obj.mixing_)
        print '\n'



# Dimensionality reduction classes using sklearn manifold package
from sklearn import manifold

class Isomap(dimenReductClass):
    # Class to interface with Isomap (Isometric Mapping) Embedding objects from scikit-learn Manifold Learning module

    def __init__(self, data, n_components=2, n_neighbors=5, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto'):
        self.n_components=n_components
        self.n_neighbors=n_neighbors 
        self.eigen_solver=eigen_solver
        self.tol=tol
        self.max_iter=max_iter
        self.path_method=path_method
        self.neighbors_algorithm=neighbors_algorithm

        self.data = data
        self.dataTransformed = None
        self.obj = manifold.Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto')

    def print_results(self): 
        print '\n Distribution Matrix'
        print(self.obj.dist_matrix_)
        print '\n Reconstruction Error'
        print(self.obj.reconstruction_error())
        print '\n'


class LocallyLinearEmbedding(dimenReductClass):
    # Class to interface with Locally Linear Embedding objects from scikit-learn Manifold Learning module

    def __init__(self, data, n_components=2, n_neighbors=5, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None):
        self.n_components=n_components
        self.n_neighbors=n_neighbors
        self.reg=reg
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
        self.obj = manifold.LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components, reg=self.reg, eigen_solver=self.eigen_solver, tol=self.tol, max_iter=self.max_iter, method=self.method, hessian_tol=self.hessian_tol, modified_tol=self.modified_tol, neighbors_algorithm=self.neighbors_algorithm, random_state=self.random_state)

    def print_results(self):
        print '\n Reconstruction Error'
        print(self.obj.reconstruction_error_)
        print '\n'


class SpectralEmbedding(dimenReductClass):
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

    def print_results(self):
        print '\n Affinity Matrix'
        print(self.obj.affinity_matrix_)
        print '\n'


