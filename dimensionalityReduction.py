import numpy as np
import visualizations as viz

# Dimensionality reduction classes using sklearn decomposition package
from sklearn import decomposition

class dimenReductClass:
    """Base class for dimensionality reduction analysis.

    The init method must always be overridden by the derived class, since each dimensionality 
    reduction technique requires its own scikit-learn object with different default input variables.
    """

    def init(self): # Should always be ovverridden by derived classes
        # May contain numerous variables for a given analysis method, but will always include:
        self.data                   ##!< Raw 2-dimensional data matrix from dataset.data
        self.dataTransformed        ##<! Primary output of clustering analysis: the index of the closest cluster
        self.obj                    ##<! The classification object for a given method from scikit-learn
                                                                        
    def fit_model(self):
        """Run the dimensionality reduction method, and return the transformed dataset (the components)

        All dimensionality reduction objects from scikit-learn have consistent fit_transform 
        methods that accept the 2-d data matrix as input and return the transformed data matrix
        (e.g. the principal components)
        """
        self.dataTransformed = self.obj.fit_transform(self.data)

    def print_results(self): # May need to be overridden by derived classes
        """Print the component matrix and the contribution of each component to the dataset variance"""
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
    """Derived class to implement Principal Component Analysis (PCA).

    Principal component anlysis is a classic statistical technique that extracts a set of orthogonal
    components from a multidimensional dataset. These components are returned in order of importance,
    such that the first principal component explains the largest share of the dataset variance, followed
    by the second largest, etc. If the number of principal components extracted is equal to the number of
    variables, no data is lost. However, even if only, say, the first two components are saved, much of
    the variability in the dataset is preserved in just these two components.

    As sklearn warns, "This implementation uses the scipy.linalg implementation of the singular value
    decomposition. It only works for dense arrays and is not scalable to large dimensional data."
    For larger arrays, try randomized PCA.

    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in PCA.obj) contains:

    Attributes: 
        components_                 (array, [n_components, n_features]) Components with maximum variance.
        explained_variance_ratio_   (array, [n_components]) Percentage of variance explained by each of the selected components.
        mean_                       (array, [n_features]) Per-feature empirical mean, estimated from the training set.
        n_components_               (int) The estimated number of components.
        noise_variance_             (float) The estimated noise covariance following the Probabilistic PCA model from Tipping and Bishop 1999.

    Methods:
        fit(X[, y])                 Fit the model with X.
        fit_transform(X[, y])       Fit the model with X and apply the dimensionality reduction on X.
        get_covariance()            Compute data covariance with the generative model.
        get_params([deep])          Get parameters for this estimator.
        get_precision()             Compute data precision matrix with the generative model.
        inverse_transform(X)        Transform data back to its original space, i.e.,
        score(X[, y])               Return the average log-likelihood of all samples
        score_samples(X)            Return the log-likelihood of each sample
        set_params(**params)        Set the parameters of this estimator.
        transform(X)                Apply the dimensionality reduction on X.
    """

    def __init__(self, data, n_components=None, copy=True, whiten=False):
        """ Create the Principal Component Analysis object.
        Initialize all provided variables and specify all default options.
        """
        self.n_components=n_components          ##!< Number of components to keep. If None, all kept (int, default: None)
        self.copy=copy                          ##!< Save data passed to fit()? (boolean, default: True) 
        self.whiten=whiten                      ##!< Ensures uncorrelated outputs with unit variance 

        self.data = data                        ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None             ##!< Output of dimensionality reduction, the dataset transformed into component space
        self.obj = decomposition.PCA(n_components=self.n_components, copy=self.copy, whiten=self.whiten) ##!< The PCA object from scikit-learn


class RPCA(dimenReductClass):
    """Derived class to implement Randomized Principal Component Analysis (RPCA).

    Faster, approximate Principal Component Analysis using randomized Singular Value Decomposition
    
    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in RPCA.obj) contains:

    Attributes: 
        components_                 (array, [n_components, n_features]) Components with maximum variance.
        explained_variance_ratio_   (array, [n_components]) Percentage of variance explained by each of the selected components.
        mean_                       (array, [n_features]) Per-feature empirical mean, estimated from the training set.

    Methods:
        fit(X[, y])                 Fit the model with X by extracting the first principal components..
        fit_transform(X[, y])       Fit the model with X and apply the dimensionality reduction on X.
        get_params([deep])          Get parameters for this estimator.
        inverse_transform(X)        Transform data back to its original space, i.e.,
        set_params(**params)        Set the parameters of this estimator.
        transform(X)                Apply the dimensionality reduction on X.
    """

    def __init__(self, data, n_components=None, copy=True, iterated_power=3, whiten=False, random_state=None):
        """ Create the Randomized PCA object.
        Initialize all provided variables and specify all default options.
        """
        self.n_components=n_components          ##!< Number of components to keep. If None, all kept (int, default: None)
        self.copy=copy                          ##!< Save data passed to fit()? (boolean, default: True) 
        self.iterated_power=iterated_power      ##!< Number of iterations for the power method (int, default: 3)
        self.whiten=whiten                      ##!< Ensures uncorrelated outputs with unit variance 
        self.random_state=random_state          ##!< Seed for RNG

        self.data = data                        ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None             ##!< Output of dimensionality reduction, the dataset transformed into component space
        self.obj = decomposition.RandomizedPCA(n_components=self.n_components, copy=self.copy, iterated_power=self.iterated_power, whiten=self.whiten, random_state=self.random_state) ##!< The Randomized PCA object from scikit-learn


class SPCA(dimenReductClass):
    """Derived class to implement Sparse Principal Component Analysis (SPCA).

    Finds sparse approximations for the principal components of a dataset by applying an L1 
    penalty to non-sparse components. May result in large errors if components are not amenable
    to a sparse representation.

    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in SPCA.obj) contains:

    Attributes: 
        components_                 (array, [n_components, n_features]) Components with maximum variance.
        error_                      (array) Vectors of errors at each iteration

    Methods:
        fit(X[, y])                 Fit the model from data in X.
        fit_transform(X[, y])       Fit to data, then transform it.
        get_params([deep])          Get parameters for this estimator.
        set_params(**params)        Set the parameters of this estimator.
        transform(X)                Least Squares projection of the data onto the sparse components.
    """

    def __init__(self, data, n_components=None, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=1, U_init=None, V_init=None, verbose=True, random_state=None):
        """ Create the Sparse PCA object.
        Initialize all provided variables and specify all default options.
        """
        self.n_components=n_components          ##!< Number of components to keep. If None, all kept (int, default: None)
        self.alpha=alpha                        ##!< Sparsity controlling parameter - higher = sparser (float, default: 1).
        self.ridge_alpha=ridge_alpha            ##!< Amount of ridge shrinkage to apply in order to improve conditioning when calling the transform method (float, default: 0.01).
        self.max_iter=max_iter                  ##!< Maximum number of iterations to perform (int, default: 1000).
        self.tol=tol                            ##!< Tolerance for the stopping condition (float, default: 1e-8).
        self.method=methods                     ##!< Options: 'lars', 'cd' (default: 'lars'). LARS = Least Angle Regression, CD = Coordinate Descent
        self.n_jobs=n_jobs                      ##!< Number of parallel jobs to run (int, default: 1)
        self.U_init=U_init                      ##!< Initial values for the loadings for warm restart scenarios
        self.V_init=V_init                      ##!< Initial values for the loadings for warm restart scenarios.
        self.verbose=verbose                    ##!< Print verbose output during calculation (int, default: true)
        self.random_state=random_state          ##!< Seed for RNG

        self.data = data                        ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None             ##!< Output of dimensionality reduction, the dataset transformed into component space
        self.obj = decomposition.SparsePCA(n_components=self.n_components, alpha=self.alpha, ridge_alpha=self.ridge_alpha, max_iter=self.max_iter, tol=self.tol, method=self.method, n_jobs=self.n_jobs, U_init=self.U_init, V_init=self.V_init, verbose=self.verbose, random_state=self.random_state) ##!< The Sparse PCA object from scikit-learn


    def print_results(self):
        print '\n Components'
        print(self.obj.components_)
        print '\n Sum squared error'
        print(np.sum(np.power(self.obj.error_,2)))
        print '\n'


class ICA(dimenReductClass):
    """Derived class to implement Independent Component Analysis (ICA).

    Like Principal Component Analysis (PCA), Independent Component Analysis acts to find a set
    of linearly independent components which can explain the variance of the dataset. However,
    unlike in contrast to the case of PCA, these components need not be orthogonal as well as
    linearly independent. ICA is particularly useful for cases of decomposing a multivariate
    signal into independent non-gaussian components (e.g. reconstructing individual sound sources
    in a room by comparing signals from several microphones).

    Unlike PCA, ICA  does not apply any relative priority to components with explain more of 
    the dataset variane.

    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in ICA.obj) contains:

    Attributes: 
        components_                     (2D array, shape (n_components, n_features)) The unmixing matrix.
        mixing_                         (array, shape (n_features, n_components)) The mixing matrix.

    Methods:
        fit(X[, y])                     Fit the model to X.
        fit_transform(X[, y])           Fit the model and recover the sources from X.
        get_params([deep])              Get parameters for this estimator.
        inverse_transform(X[, copy])    Transform the sources back to the mixed data (apply mixing matrix).
        set_params(**params)            Set the parameters of this estimator.
        transform(X[, y, copy])         Recover the sources from X (apply the unmixing matrix).
    """

    def __init__(self, data, n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, w_init=None, random_state=None):
        """ Create the ICA object.
        Initialize all provided variables and specify all default options.
        """
        self.n_components=n_components          ##!< Number of components to keep. If None, all kept (int, default: None)
        self.algorithm=algorithm                ##!< Options: 'parallel', 'deflation' (default: 'parallel'). From FastICA.
        self.whiten=whiten                      ##!< Ensures uncorrelated outputs with unit variance 
        self.fun=fun                            ##!< Options: 'logcosh', 'exp', 'cube', or a callable function (default: 'logcosh'). The functional form of the G function used in the approximation to neg-entropy.
        self.fun_args=fun_args                  ##!< Arguments to send to the G function
        self.max_iter=max_iter                  ##!< Maximum number of iterations to perform (int, default: 200).
        self.tol=tol                            ##!< Tolerance on update at each iteration (default: 1e-4).
        self.w_init=w_init                      ##!< The mixing matrix to be used to initialize the algorithm (default: None).
        self.random_state=random_state          ##!< Seed for RNG

        self.data = data                        ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None             ##!< Output of dimensionality reduction, the dataset transformed into component space
        self.obj = decomposition.FastICA(n_components=self.n_components, algorithm=self.algorithm, whiten=self.whiten, fun=self.fun, fun_args=self.fun_args, max_iter=self.max_iter, tol=self.tol, w_init=self.w_init, random_state=self.random_state) ##!< The ICA object from scikit-learn


    def print_results(self):
        print '\n Mixing matrix'
        print(self.obj.mixing_)
        print '\n'



# Dimensionality reduction classes using sklearn manifold package
from sklearn import manifold

class Isomap(dimenReductClass):
    """Derived class to implement Isomap (Isometric Mapping) Embedding.

    "Non-linear dimensionality reduction through Isometric Mapping"

    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in PCA.obj) contains:

    Attributes: 
        embedding_                      (array-like, shape [n_samples, n_components]) Stores the embedding vectors.
        kernel_pca_                     (object) KernelPCA object used to implement the embedding.
        training_data_                  (array-like, shape [n_samples, n_features]) Stores the training data.
        nbrs_                           (sklearn.neighbors.NearestNeighbors instance) Stores nearest neighbors instance, including BallTree or KDtree if applicable.
        dist_matrix_                    (array-like, shape [n_samples, n_samples]) Stores the geodesic distance matrix of training data.

    Methods:
        fit(X[, y])                     Compute the embedding vectors for data X
        fit_transform(X[, y])           Fit the model from data in X and transform X.
        get_params([deep])              Get parameters for this estimator.
        reconstruction_error()          Compute the reconstruction error for the embedding.
        set_params(**params)            Set the parameters of this estimator.
        transform(X)                    Transform X.
    """

    def __init__(self, data, n_components=2, n_neighbors=5, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto'):
        """ Create the Isomap Embedding object.
        Initialize all provided variables and specify all default options.
        """
        self.n_components=n_components          ##!< Number of components to keep (int, default: 2)
        self.n_neighbors=n_neighbors            ##!< Number of neighbors to consider at each point (int, default: 5).
        self.eigen_solver=eigen_solver          ##!< Options for eigenvalue problem: 'auto', 'arpack', 'dense' (default 'auto'). ARPACK = Arnoldi decomposition, dense = direct solver
        self.tol=tol                            ##!< Convergence tolerance passed to arpack or lobpcg (float, default: 0)
        self.max_iter=max_iter                  ##!< Maximum number of iterations if using the 'arpack' solver.
        self.path_method=path_method            ##!< Options for finding shortest path: 'auto','FW','D' (default: 'auto'). FW = Floyd-Warshall algorithm, D = Dijkstra algorithm with Fibonacci Heaps
        self.neighbors_algorithm=neighbors_algorithm ##!< Algorithm for nearest neighbors search, passed to sklearn.neighbors.NearestNeighbors. Options: 'auto', 'brute','kd_tree','ball_tree' (default: 'auto).

        self.data = data                        ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None             ##!< Output of dimensionality reduction, the dataset transformed into component space
        self.obj = manifold.Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto') ##!< The Isomap object from scikit-learn

    def print_results(self): 
        print '\n Distribution Matrix'
        print(self.obj.dist_matrix_)
        print '\n Reconstruction Error'
        print(self.obj.reconstruction_error())
        print '\n'


class LocallyLinearEmbedding(dimenReductClass):
    """Derived class to implement Locally Linear Embedding.

    Non-linear dimensionality reduction through locally linear mapping.

    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in PCA.obj) contains:

    Attributes: 
        embedding_vectors_              (array-like, shape [n_components, n_samples]) Stores the embedding vectors
        reconstruction_error_           (float) Reconstruction error associated with embedding_vectors_
        nbrs_                           (NearestNeighbors object) Stores nearest neighbors instance, including BallTree or KDtree if applicable.

    Methods:
        fit(X[, y])                     Compute the embedding vectors for data X
        fit_transform(X[, y])           Compute the embedding vectors for data X and transform X.
        get_params([deep])              Get parameters for this estimator.
        set_params(**params)            Set the parameters of this estimator.
        transform(X)                    Transform new points into embedding space.
    """

    def __init__(self, data, n_components=2, n_neighbors=5, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None):
        """ Create the Locally Linear Embedding object.
        Initialize all provided variables and specify all default options.
        """
        self.n_components=n_components          ##!< Number of components to keep (int, default: 2)
        self.n_neighbors=n_neighbors            ##!< Number of neighbors to consider at each point (int, default: 5).
        self.reg=reg                            ##!< regularization constant, multiplies the trace of the local covariance matrix of the distances (int, default: 0.001).
        self.eigen_solver=eigen_solver          ##!< Options for eigenvalue problem: 'auto', 'arpack', 'dense' (default 'auto'). ARPACK = Arnoldi decomposition, dense = direct solver
        self.tol=tol                            ##!< Tolerance passed to arpack eigensolver, if used (float, default 1e-6).
        self.max_iter=max_iter                  ##!< Maximum number of iterations to perform (int, default: 100).
        self.method=method                      ##!< Options: 'standard', 'hessian' (hessian eigenmapping),'modified, or 'ltsa' (local tangent space alignment). Default: 'standard'
        self.hessian_tol=hessian_tol            ##!< Tolerance for hessian eigenmapping method (float, default: 1e-4).
        self.modified_tol=modified_tol          ##!< Tolerance for modified LLE method (float, default 1e-12).
        self.neighbors_algorithm=neighbors_algorithm ##!< Algorithm for nearest neighbors search, passed to sklearn.neighbors.NearestNeighbors. Options: 'auto', 'brute','kd_tree','ball_tree' (default: 'auto).
        self.random_state=random_state          ##!< Seed for RNG

        self.data = data                        ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None             ##!< Output of dimensionality reduction, the dataset transformed into component space
        self.obj = manifold.LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components, reg=self.reg, eigen_solver=self.eigen_solver, tol=self.tol, max_iter=self.max_iter, method=self.method, hessian_tol=self.hessian_tol, modified_tol=self.modified_tol, neighbors_algorithm=self.neighbors_algorithm, random_state=self.random_state) ##!< The Locally Linear Embedding object from scikit-learn


    def print_results(self):
        print '\n Reconstruction Error'
        print(self.obj.reconstruction_error_)
        print '\n'


class SpectralEmbedding(dimenReductClass):
    """Derived class to implement Spectral Embedding.

    Non-linear dimensionality reduction through spectral embedding.

    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in PCA.obj) contains:

    Attributes: 
        embedding_                      (array, shape = [n_samples, n_components]) Spectral embedding of the training matrix.
        affinity_matrix_                (array, shape = [n_samples, n_samples]) Affinity_matrix constructed from samples or precomputed.

    Methods:
        fit(X[, y])                     Fit the model from data in X.
        fit_transform(X[, y])           Fit the model from data in X and transform X.
        get_params([deep])              Get parameters for this estimator.
        set_params(**params)            Set the parameters of this estimator.
    """

    def __init__(self, data, n_components=2, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, n_neighbors=None):
        """ Create the Spectral Embedding object.
        Initialize all provided variables and specify all default options.
        """
        self.n_components=n_components          ##!< Number of components to keep (int, default: 2)
        self.affinity=affinity                  ##!< How to construct the affinity matrix. Options: 'nearest neighbors' (knn), 'rbf', 'precomputed', or a callable function (default: 'nearest neighbors').
        self.gamma=gamma                        ##!< Kernel coefficient for rbf kernel (float, default: None).
        self.random_state=random_state          ##!< Seed for RNG
        self.eigen_solver=eigen_solver          ##!< Options for eigenvalue decomposition: None, 'arpack', 'lobpcg', 'amg' (default: None). AMG requires pyamg.
        self.n_neighbors=n_neighbors            ##!< Number of neighbors to consider at each point (int, default: 5).

        self.data = data                        ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None             ##!< Output of dimensionality reduction, the dataset transformed into component space
        self.obj = manifold.SpectralEmbedding(n_components=self.n_components, affinity=self.affinity, gamma=self.gamma, random_state=self.random_state, eigen_solver=self.eigen_solver, n_neighbors=self.n_neighbors) ##!< The Spectral Embedding object from scikit-learn


    def print_results(self):
        print '\n Affinity Matrix'
        print(self.obj.affinity_matrix_)
        print '\n'


