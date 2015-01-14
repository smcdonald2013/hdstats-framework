from sklearn import cluster
import visualizations as viz


class clusterClass:
    """Base class for cluster analysis.
       
    The init method must always be overridden by the derived class, since each clustering 
    technique requires its own scikit-learn object with different default input variables.
    """

    def init(self): # Should always be ovverridden
        # May contain numerous variables for a given analysis method, but will always include:
        self.data 
        self.dataTransformed 
        self.obj 
        

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


class KMeans(clusterClass):

    def __init__(self, data, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=1, random_state=None, n_jobs=-1):
        self.n_clusters=n_clusters      
        self.max_iter=max_iter          
        self.n_init=n_init              
        self.init=init                  
        self.tol=tol                    
        self.precompute_distances=precompute_distances 
        self.verbose=verbose            
        self.random_state=random_state  
        self.n_jobs=n_jobs              


        self.data = data                
        self.dataTransformed = None     
        self.obj = cluster.KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances, verbose=self.verbose, random_state=self.random_state, n_jobs=self.n_jobs) 



class MiniBatchKMeans:

    def __init__(self, data, n_clusters=8, init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01):
        self.n_clusters=n_clusters          
        self.max_iter=max_iter              
        self.init=init                      
        self.batch_size=batch_size          
        self.verbose=verbose                
        self.compute_labels=compute_labels  
        self.random_state=random_state      
        self.tol=tol                        
        self.max_no_improvement=max_no_improvement 
        self.init_size=init_size            
        self.n_init=n_init                  
        self.reassignment_ratio=reassignment_ratio 

        self.data = data                    
        self.dataTransformed = None         
        self.obj = cluster.MiniBatchKMeans(n_clusters=self.n_clusters, init=self.init, max_iter=self.max_iter, batch_size=self.batch_size, verbose=self.verbose, compute_labels=self.compute_labels, random_state=self.random_state, tol=self.tol, max_no_improvement=self.max_no_improvement, init_size=self.init_size, n_init=self.n_init, reassignment_ratio=self.reassignment_ratio) 


class MeanShift:


    def __init__(self, data, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True):
        self.bandwidth=bandwidth        
        self.seeds=seeds                
        self.bin_seeding=bin_seeding    
        self.min_bin_freq=min_bin_freq  
        self.cluster_all=cluster_all    

        self.data = data                
        self.dataTransformed = None     
        self.obj = cluster.MeanShift(bandwidth=self.bandwidth, seeds=self.seeds, bin_seeding=self.bin_seeding, min_bin_freq=self.min_bin_freq, cluster_all=self.cluster_all) 


class SpectralClustering:

    def __init__(self, data, n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None):
        self.n_clusters=n_clusters          
        self.eigen_solver=eigen_solver      
        self.random_state=random_state      
        self.n_init=n_init                  
        self.gamma=gamma                    
        self.affinity=affinity              
        self.n_neighbors=n_neighbors        
        self.eigen_tol=eigen_tol            
        self.assign_labels=assign_labels    
        self.degree=degree                  
        self.coef0=coef0                    
        self.kernel_params=kernel_params    

        self.data = data                    
        self.dataTransformed = None         
        self.obj = cluster.SpectralClustering(n_clusters=self.n_clusters, eigen_solver=self.eigen_solver, random_state=self.random_state, n_init=self.n_init, gamma=self.gamma, affinity=self.affinity, n_neighbors=self.n_neighbors, eigen_tol=self.eigen_tol, assign_labels=self.assign_labels, degree=self.degree, coef0=self.coef0, kernel_params=self.kernel_params) 

    def print_results(self):
        pass


class DBSCAN:

    def __init__(self, data, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None):
        self.eps=eps                        
        self.min_samples=min_samples        
        self.metric=metric                  
        self.algorithm=algorithm            
        self.leaf_size=leaf_size            
        self.p=p                            
        self.random_state=random_state      

        self.data = data                    
        self.dataTransformed = None         
        self.obj = cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, random_state=self.random_state) 


    def print_results(self):
        pass

