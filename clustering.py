from sklearn import cluster

class KMeans:
    # Class to interface with KMeans Clustering object from scikit-learn cluster module

    def __init__(self, data, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1):
        self.n_clusters=n_clusters
        self.init=init
        self.n_init=n_init
        self.max_iter=max_iter
        self.tol=tol
        self.precompute_distances=precompute_distances
        self.verbose=verbose
        self.random_state=random_state
        self.copy_x=copy_x
        self.n_jobs=n_jobs

        self.data = data
        self.result = None
        self.dataTransformed = None
        self.obj = cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=1, random_state=None, copy_x=True, n_jobs=1)

    def fit_model(self):
        self.dataTransformed = self.obj.fit_transform(self.data) # Transformed to cluster-distance space
        self.result = self.obj.fit_predict(self.data) # Index of closest cluster

    def print_results(self):
        print('\n Cluster Centers')
        print(self.obj.cluster_centers_)
        print('\n')

    def plot_results(self):
        # plot cluster centers along with all data, colored by nearest cluster
        pass


