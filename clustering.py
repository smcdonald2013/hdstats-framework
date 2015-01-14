from sklearn import cluster
import visualizations as viz


class clusterClass:
    """Base class for cluster analysis.
       
    The init method must always be overridden by the derived class, since each clustering 
    technique requires its own scikit-learn object with different default input variables.
    """

    def init(self): # Should always be ovverridden
        # May contain numerous variables for a given analysis method, but will always include:
        self.data ##!< Raw 2-dimensional data matrix from dataset.data
        self.dataTransformed ##<! Primary output of clustering analysis: the index of the closest cluster
        self.obj ##<! The classification object for a given method from scikit-learn
        

    def fit_model(self):
        """Run the cluster analysis method, and return the list of closest cluster indices.

        All clustering objects from scikit-learn have consistent fit_predict methods that
        accept the 2-d data matrix as input and return the cluster index of each row.
        """
        self.dataTransformed = self.obj.fit_predict(self.data) 

    def print_results(self):
        """Print the list of closest cluster indices, which is the common output of all clustering methods"""
        print '\n Cluster Centers'
        print(self.obj.cluster_centers_)
        print '\n'

    def plot_results(self):
        """Plot the first two variables in the dataset, colored by cluster index.
        Assuming the dataset has been run through a dimensionality reduction technique first, 
        these first two variables should ideally capture  most of the variance in the dataset.

        Uses plot_clusters class from visualizations.py.
        """
        viz.plot_clusters(self.data[:,0], self.data[:,1], self.dataTransformed).plot()


# 
class KMeans(clusterClass):
    """Derived class to implement KMeans Clustering.

    This is arguably the "default" clustering method. As the name suggests, it attempts to 
    separate however many samples it is provided with into n clusters based on cluster means,
    such that each sample belongs to the cluster with the closest mean.

    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in KMeans.obj) contains:

    Attributes: 
        cluster_centers_            (array, [n_clusters, n_features]) Coordinates of cluster centers
        labels_                     (list) Labels of each point
        inertia_                    (float) Sum of distances of samples to their closest cluster center.

    Methods:
        fit(X[, y])                 Compute k-means clustering.
        fit_predict(X)              Compute cluster centers and predict cluster index for each sample.
        fit_transform(X[, y])       Compute clustering and transform X to cluster-distance space.
        get_params([deep])          Get parameters for this estimator.
        predict(X)                  Predict the closest cluster each sample in X belongs to.
        score(X)                    Opposite of the value of X on the K-means objective.
        set_params(**params)        Set the parameters of this estimator.
        transform(X[, y])           Transform X to a cluster-distance space.
    """

    def __init__(self, data, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=1, random_state=None, n_jobs=-1):
        """ Create the KMeans Clustering object.
        Initialize all provided variables and specify all default options.
        """
        self.n_clusters=n_clusters      ##!< The number of clusters to form as well as the number of centroids to generate (int, default: 8).
        self.max_iter=max_iter          ##!< Maximum number of iterations of the k-means algorithm for a single run (int, default: 300).
        self.n_init=n_init              ##!< Number of time the k-means algorithm will be run with different centroid seeds (int, default: 10).
                                        # The final results will be the best output of n_init consecutive runs in terms of inertia.
        self.init=init                  ##!< Method for initialization. Options: 'kmeans++', 'random', or ndarray (default 'k-means++')
                                        # kmeans++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
                                        # random: Choose k observations (rows) at random from data for the initial centroids.
                                        # If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        self.tol=tol                    ##!< Relative tolerance with regards to inertia to declare convergence (float, default: 1e-4)
        self.precompute_distances=precompute_distances ##!< Faster, but takes more memory (boolean, default: True)
        self.verbose=verbose            ##!< Print verbose output during calculation (boolean, default: True)
        self.random_state=random_state  ##!< Seed for RNG used to initialize the centers. If None, use global NumPy default (int, default: None)
        self.n_jobs=n_jobs              ##!< Number of tasks to use in parallel calculation. If -1, one for each cpu is used (int, default: -1)


        self.data = data                ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None     ##!< Output of clustering analysis, the list of cluster indices for each sample.
        self.obj = cluster.KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances, verbose=self.verbose, random_state=self.random_state, n_jobs=self.n_jobs) ##!< The KMeans clustering object from scikit-learn



class MiniBatchKMeans(clusterClass):
    """Derived class to implement Mini-Batch KMeans Clustering.

    A modification of standard KMeans clustering with better scalability

    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in MiniBatchKMeans.obj) contains:

    Attributes: 
        cluster_centers_            (array, [n_clusters, n_features]) Coordinates of cluster centers
        labels_                     (list) Labels of each point
        inertia_                    (float) Sum of distances of samples to their closest cluster center.

    Methods:
        fit(X[, y])                 Compute the centroids on X by chunking it into mini-batches.
        fit_predict(X)              Compute cluster centers and predict cluster index for each sample.
        fit_transform(X[, y])       Compute clustering and transform X to cluster-distance space.
        get_params([deep])          Get parameters for this estimator.
        partial_fit(X[, y])         Update k means estimate on a single mini-batch X.
        predict(X)                  Predict the closest cluster each sample in X belongs to.
        score(X)                    Opposite of the value of X on the K-means objective.
        set_params(**params)        Set the parameters of this estimator.
        transform(X[, y])           Transform X to a cluster-distance space.
    """

    def __init__(self, data, n_clusters=8, init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01):
        self.n_clusters=n_clusters          ##!< The number of clusters to form as well as the number of centroids to generate (int, default: 8). 
        self.max_iter=max_iter              ##!< Maximum number of iterations of the k-means algorithm for a single run (int, default: 100).
        self.init=init                      ##!< Method for initialization. Options: 'kmeans++', 'random', or ndarray (default 'k-means++')
                                            # kmeans++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
                                            # random: Choose k observations (rows) at random from data for the initial centroids.
                                            # If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers
        self.batch_size=batch_size          ##!< Size of the mini-batches (int, default: 100).
        self.verbose=verbose                ##!< Print verbose output during calculation (boolean, default: True).
        self.compute_labels=compute_labels  ##!< Compute label assignment and inertia for the complete dataset once the minibatch optimization has converged (boolean, default: True).
        self.random_state=random_state      ##!< Seed for RNG used to initialize the centers. If None, use global NumPy default (int, default: None).
        self.tol=tol                        ##!< Tolerance for convergence detection based on normalized center change. Not used if 0 (float, default 0.0).
        self.max_no_improvement=max_no_improvement ##!< Stop after this consecutive number of mini batches that does not yield an improvement on the smoothed inertia (int, default: 10).
        self.init_size=init_size            ##!< Number of samples to randomly sample for speeding up the initialization (int, default: 3*batch_size). Must be larger than k.
        self.n_init=n_init                  ##!< Number of random initializations that are tried (int, default: 3). 
                                            # In contrast to KMeans, the algorithm is only run once, using the best of the n_init initializations as measured by inertia.
        self.reassignment_ratio=reassignment_ratio ##!< The fraction of the maximum number of counts for a center to be reassigned. 
                                            # A higher value means that low count centers are more easily reassigned, which means that
                                            # the model will take longer to converge, but should converge in a better clustering.

        self.data = data                    ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None         ##!< Output of clustering analysis, the list of cluster indices for each sample.
        self.obj = cluster.MiniBatchKMeans(n_clusters=self.n_clusters, init=self.init, max_iter=self.max_iter, batch_size=self.batch_size, verbose=self.verbose, compute_labels=self.compute_labels, random_state=self.random_state, tol=self.tol, max_no_improvement=self.max_no_improvement, init_size=self.init_size, n_init=self.n_init, reassignment_ratio=self.reassignment_ratio) ##!< The MiniBatch KMeans clustering object from scikit-learn


class MeanShift(clusterClass):
    """Derived class to implement MeanShift clustering.

    Unlike KMeans-based and spectral clustering types, Mean-shift Clustering does not need
    to know the number of clusters to find a priori. Mean-shift clustering is a non-parametric
    method that looks for areas of high sample density and identifies candidate centroids.
    Candidates that are overly close together are then eliminated. Scalability can be improved
    by using fewer seeds. However if bandwidth is not provided, bandwidth estimation can become
    the limiting step.
    
    Variable descriptions below are those of the underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in MeanShift.obj) contains:

    Attributes: 
        cluster_centers_            (array, [n_clusters, n_features]) Coordinates of cluster centers
        labels_                     (list) Labels of each point

    Methods:
        fit(X)                      Perform clustering.
        fit_predict(X[, y])         Performs clustering on X and returns cluster labels.
        get_params([deep])          Get parameters for this estimator.
        predict(X)                  Predict the closest cluster each sample in X belongs to.
        set_params(**params)        Set the parameters of this estimator.
    """

    def __init__(self, data, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True):
        self.bandwidth=bandwidth        ##!< Bandwidth used in the RBF kernel (float, default: None). If None, estimated by sklearn.cluster.estimate_bandwitdth
        self.seeds=seeds                ##!< Seeds used to initialize kernels (array, default: None). If None, calculated by sklearn.clustering.get_bin_seeds 
        self.bin_seeding=bin_seeding    ##!< Seed on a grid scaled by bandwidth (boolean, default: False).
        self.min_bin_freq=min_bin_freq  ##!< Accept only bins with at leas this many seeds (int, default: 1). Increase for faster calculation.
        self.cluster_all=cluster_all    ##!< Cluster points even if they don't fall within a kernel (boolean, default: True).

        self.data = data                ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None     ##!< Output of clustering analysis, the list of cluster indices for each sample.
        self.obj = cluster.MeanShift(bandwidth=self.bandwidth, seeds=self.seeds, bin_seeding=self.bin_seeding, min_bin_freq=self.min_bin_freq, cluster_all=self.cluster_all) ##!< The MeanShift clustering object from scikit-learn


class SpectralClustering(clusterClass):
    """Derived class to implement Spectral Clustering.
 
    Spectral Clustering is a clustering technique that uses the eigenvalues of the similarity 
    matrix of the dataset. It is considered particularly good at finding non-convex clusters.
        
    Variable descriptions below are those of using underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in MeanShift.obj) contains:

    Attributes: 
        affinity_matrix_            (array-like, [n_clusters, n_features]) Affinity matrix used for clustering.
        labels_                     (list) Labels of each point

    Methods:
        fit(X)                      Creates an affinity matrix for X using the selected affinity, then applies spectral clustering to this affinity matrix.
        fit_predict(X[, y])         Performs clustering on X and returns cluster labels.
        get_params([deep])          Get parameters for this estimator.
        set_params(**params)        Set the parameters of this estimator.
    """

    def __init__(self, data, n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None):
        self.n_clusters=n_clusters          ##!< The number of clusters to form as well as the number of centroids to generate (int, default: 8). 
        self.eigen_solver=eigen_solver      ##!< The eigenvalue decomposition strategy to use. Options: None, 'arpack', 'lobpcg', and 'amg' (default: None). AMG requires pyamg.
        self.random_state=random_state      ##!< RNG seed. Used for AMG method and for K-Means initialization
        self.n_init=n_init                  ##!< For k-means initialization. Number of times algorithm will be run with different centroid seeds
        self.gamma=gamma                    ##!< Scaling factor of RBF, polynomial, exponential chi^2 and sigmoid affinity kernel (float, default: 1.0)
        self.affinity=affinity              ##!< Options: 'nearest_neighbors', 'precomputed', 'rbf' or one of the kernels supported by sklearn.metrics.pairwise_kernels.
        self.n_neighbors=n_neighbors        ##!< Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method (int, default: 10).
        self.eigen_tol=eigen_tol            ##!< Stopping criterion for eigendecomposition of the Laplacian matrix when using arpack eigen_solver (float, default 0.0).
        self.assign_labels=assign_labels    ##!< Options: 'kmeans', 'discretize' (default: 'kmeans'). Method for assigining initial lables in embedding space.
        self.degree=degree                  ##!< Degree if using polynomial kernel (float, default: 3).
        self.coef0=coef0                    ##!< Zero coefficient if using polynomial or sigmoid kernel (float, default: 1).
        self.kernel_params=kernel_params    ##!< Parameters and values if using a kernel passed as a callable object.

        self.data = data                    ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None         ##!< Output of clustering analysis, the list of cluster indices for each sample.
        self.obj = cluster.SpectralClustering(n_clusters=self.n_clusters, eigen_solver=self.eigen_solver, random_state=self.random_state, n_init=self.n_init, gamma=self.gamma, affinity=self.affinity, n_neighbors=self.n_neighbors, eigen_tol=self.eigen_tol, assign_labels=self.assign_labels, degree=self.degree, coef0=self.coef0, kernel_params=self.kernel_params) ##!< The spectral clustering object from scikit-learn

    def print_results(self):
        pass


class DBSCAN(clusterClass):
    """Derived class to implement DBSCAN clustering.
 
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering 
    technique which finds areas of high sample density (like MeanShift) and then works outward
    to define clusters
    
            
    Variable descriptions below are those of using underlying scikit-learn clustering object.
    For reference, the underlying scikit object (in MeanShift.obj) contains:

    Attributes: 
        affinity_matrix_            (array-like, [n_clusters, n_features]) Affinity matrix used for clustering.
        labels_                     (list) Labels of each point

    Methods:
        fit(X)                      Creates an affinity matrix for X using the selected affinity, then applies spectral clustering to this affinity matrix.
        fit_predict(X[, y])         Performs clustering on X and returns cluster labels.
        get_params([deep])          Get parameters for this estimator.
        set_params(**params)        Set the parameters of this estimator.
    """
    def __init__(self, data, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None):
        self.eps=eps                        ##!< The maximum distance between two samples for them to be considered as in the same neighborhood (float, default: 0.5).
        self.min_samples=min_samples        ##!< The number of samples in a neighborhood for a point to be considered as a core point (int, default: 5).
        self.metric=metric                  ##!< String or callable. Must be one of the options allowed by sklearn.metrics.pairwise.calculate_distance for its metric parameter.
        self.algorithm=algorithm            ##!< The algorithm to be used by the NearestNeighbors module. Options: 'auto', 'ball_tree', 'kd_tree', 'brute' (default: 'auto'). 
        self.leaf_size=leaf_size            ##!< Leaf size passed to BallTree or cKDTree (int, default: 30)
        self.p=p                            ##!< The power of the Minkowski metric to be used to calculate distance between points.
        self.random_state=random_state      ##!< Seed for RNG used to initizlize the centers

        self.data = data                    ##!< 2-d data matrix from imported dataset
        self.dataTransformed = None         ##!< Output of clustering analysis, the list of cluster indices for each sample.
        self.obj = cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, random_state=self.random_state) ##!< The DBSCAN object from scikit-learn


    def print_results(self):
        pass

