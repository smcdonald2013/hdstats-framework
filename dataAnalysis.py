import sklearn # scikit-learn
import numpy as np
import ols
import sgd
import lda
import pca
import lasso
import elasticnet
import ridge
import lars
import omp
import clustering
import logistic
import qda

def main(dataset):
    while True:
        s=raw_input('Select data analysis task:\n  1. Regression\n  2. Dimensionality reduction\n  3. Clustering\n  4. Classification\n- 0. Exit\n')
        if s=='0' or s=='': # default
            break
        elif s=='1':
            dataset=regression(dataset)
        elif s=='2':
            dataset=dimensionalityReduction(dataset)
        elif s=='3':
            dataset=clusteringAnalysis(dataset)
        elif s=='4':
            dataset=classification(dataset)
        else:
            print 'Input not recognized\n'
    return dataset


def clusteringAnalysis(dataset):
    s=raw_input('Select clustering method:\n  1. KMeans Clustering \n- 2. MiniBatch KMeans Clustering\n  3. MeanShift\n  4. Spectral Clustering\n  5. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)\n  6. Guide me\n')

    if s=='1':
        # KMeans Clustering
        s1=raw_input('Number of clusters to find? (default: 8)\n')
        try: n=int(s1)
        except: n=8
        model=clustering.KMeans(dataset.data[:,np.arange(dataset.data.shape[1]) != dataset.independentVariable],n_clusters=n)

    elif s=='2' or s=='': # default
        # MiniBatchKMeans Clustering
        s1=raw_input('Number of clusters to find? (default: 8)\n')
        try: n=int(s1)
        except: n=8
        model=clustering.MiniBatchKMeans(dataset.data[:,np.arange(dataset.data.shape[1]) != dataset.independentVariable],n_clusters=n)

    elif s=='3':
        # MeanShift Clustering
        model=clustering.MeanShift(dataset.data)

    elif s=='4':
        # Spectral Clustering
        s1=raw_input('Number of clusters to find? (default: 8)\n')
        try: n=int(s1)
        except: n=8
        model=clustering.SpectralClustering(dataset.data[:,np.arange(dataset.data.shape[1]) != dataset.independentVariable],n_clusters=n)

    elif s=='5':
        # DBSCAN
        s1=raw_input('Neighborhood size? (default: 0.5)\n')
        try: n=float(s1)
        except: n=0.5
        model=clustering.DBSCAN(dataset.data[:,np.arange(dataset.data.shape[1]) != dataset.independentVariable], eps=n)

    elif s=='6':
        # Guided Clustering
        s=raw_input('Number of clusters, if known?\n')
        try: n=int(s)
        except: n=0

        if n>0 & dataset.data.shape[0] < 10000:
            print 'Known number of clusters and <10k samples: Using KMeans clustering\n'
            # Use KMeans clustering
            model=clustering.KMeans(dataset.data[:,np.arange(dataset.data.shape[1]) != dataset.independentVariable],n_clusters=n)
        elif n>0:
            print 'Known number of clusters and >10k samples: Using MiniBatch KMeans\n'
            # Use MiniBatch KMeans
            model=clustering.MiniBatchKMeans(dataset.data[:,np.arange(dataset.data.shape[1]) != dataset.independentVariable],n_clusters=n)
        elif dataset.data.shape[0] < 10000:
            print 'Unknown number of clusters and <10k samples: Using MeanShift\n'
            # Use MeanShift
            model=clustering.MeanShift(dataset.data[:,np.arange(dataset.data.shape[1]) != dataset.independentVariable])
        else:
            print 'Too many samples to analyze without knowing number of categories\n'

    try: model.fit_model()
    except Exception, e: print 'Error fitting model: %s' % e
    try: model.print_results()
    except Exception, e: print 'Error: %s' % e
    try: model.plot_results()
    except Exception, e: print 'Error: %s' % e


    return dataset


def dimensionalityReduction(dataset):
    s=raw_input('Select dimensionality reduction method:\n  1. Principal Component Analysis (PCA)\n- 2. Randomized PCA (faster)\n  3. Sparse PCA (finds sparse principal components)\n  4. Independent Component Analysis (ICA - components need not be orthogonal)\n  5. Isometric Mapping (Isomap)\n  6. Locally Linear Embedding\n  7. Spectral Embedding\n  8. Guide me\n')

    if s=='1':
        s1=raw_input('Number of components to keep? (default: all)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        model=pca.PCA(dataset.data,n_components=n_components)

    elif s=='2' or s=='': # default
        s1=raw_input('Number of components to keep? (default: all)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        model=pca.RPCA(dataset.data,n_components=n_components)

    elif s=='3':
        s1=raw_input('Number of sparse atoms to extract? (default: number of variables)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        s1=raw_input('Alpha? (Higher = more sparse. default: 1)\n')
        try: alpha=float(s1)
        except: alpha=1

        model=pca.SPCA(dataset.data,n_components=n_components, alpha=alpha)

    elif s=='4':
        s1=raw_input('Number of components to use? (default: number of variables)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        model=pca.ICA(dataset.data,n_components=n_components)

    elif s=='5':
        s1=raw_input('Number of components to use? (default: 2)\n')
        try: n_components=int(s1)
        except: n_components=2

        s1=raw_input('Number of neighbors to consider for each point.? (default: 5)\n')
        try: n_neighbors=int(s1)
        except: n_neighbors=5

        model=pca.Isomap(dataset.data, n_components=n_components, n_neighbors=n_neighbors)

    elif s=='6':
        s1=raw_input('Number of components to use? (default: 2)\n')
        try: n_components=int(s1)
        except: n_components=2

        s1=raw_input('Number of neighbors to consider for each point.? (default: 5)\n')
        try: n_neighbors=int(s1)
        except: n_neighbors=5

        model=pca.LocallyLinearEmbedding(dataset.data, n_components=n_components, n_neighbors=n_neighbors)
    elif s=='7':
        s1=raw_input('Number of components to use? (default: 2)\n')
        try: n_components=int(s1)
        except: n_components=2

        s1=raw_input('Number of neighbors to consider for each point.? (default: 5)\n')
        try: n_neighbors=int(s1)
        except: n_neighbors=5

        model=pca.SpectralEmbedding(dataset.data, n_components=n_components, n_neighbors=n_neighbors)

    elif s=='8':
        s1=raw_input('Do you want to extract components that are orthogonal to each other?\n 0. No\n-1. Yes\n')
        s2=raw_input('How many components do you want to find? (default: number of variables)\n')
        try: n_components=int(s2)
        except: n_components=dataset.data.shape[1]

        if s1=='0':
            print 'Trying ICA'
            model=pca.ICA(dataset.data,n_components=n_components)
        else:
            print 'Trying Randomized PCA'
            model=pca.RPCA(dataset.data,n_components=n_components)


    try: model.fit_model()
    except Exception, e: print 'Error fitting model: %s' % e
    try: model.print_results()
    except Exception, e: print 'Error: %s' % e
    try: model.plot_results()
    except Exception, e: print 'Error: %s' % e

    s=raw_input('Replace dataset with transformed dataset?\n- 0. No\n  1. Yes\n')
    if s=='1':
        dataset.data = model.dataTransformed
    
    return dataset
 
def regression(dataset):
    s=raw_input('Select regression technique:\n- 1. OLS\n  2. Lasso\n  3. Ridge\n  4. Elastic Net\n  5. Lars\n  6. OMP\n  0. Guide me\n')

    if s=='1' or s=='': # default
        model=ols.OLS(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0]) #independent variable is assumed to be in the first column

    elif s=='2':
        s1=raw_input('Value of alpha parameter? (default: 1)\n')
        try: alpha=int(s1)
        except: alpha=1

        model=lasso.LASSO(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], alpha=alpha)

    elif s=='3':
        s1=raw_input('Value of alpha parameter? (default: 1)\n')
        try: alpha=int(s1)
        except: alpha=1

        model=ridge.RIDGE(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], alpha=alpha)

    elif s=='4':
        s1=raw_input('Value of alpha parameter? (default: 1)\n')
        try: alpha=int(s1)
        except: alpha=1

        s1=raw_input('Ratio of l1 penalty relative to l2? (default: .5)\n')
        try: l1_ratio = int(s1)
        except: l1_ratio=.5

        model=elasticnet.ELASTICNET(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], alpha=alpha, l1_ratio=l1_ratio)

    elif s=='5':
        s1=raw_input('Value of alpha parameter? (default: 1)\n')
        try: alpha=int(s1)
        except: alpha=1

        s1=raw_input('Ratio of l1 penalty relative to l2? (default: .5)\n')
        try: l1_ratio = int(s1)
        except: l1_ratio=.5

        model=lars.LARS(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], alpha=alpha, l1_ratio=l1_ratio)

    elif s=='6':

        model=omp.OMP(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0])

    elif s=='0':
        sp = raw_input('Is the underlying model assumed to be sparse? (Default is no)\n  1. Yes\n- 0. No\n')
        if sp=='1':
            spVal = True
        else:
            spVal = False
        if dataset.data.shape[0] < 100000:
            print('\nRunning OLS Regression')
            model = ols.OLS(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], sparse=spVal) #independent variable is assumed to be in the first column
            model.fit_model()
            print('\nRegression summary', model.fitted_model.coef_)
            print('\nPerforming assumption checks')
            model.check_model()
            print('\nTaking any necessary corrective actions')
            model.mcAction()
            model.acAction()
            model.linAction()
            model.singAction()
            model.homoskeAction()
        else:
            print('\nDataset is very large, running SGD Regression')
            model = sgd.SGD(dataset.data[:,1:dataset.data.shape[1]], dataset.data[:,0])


    try: model.fit_model()
    except Exception, e: print 'Error fitting model: %s' % e
    try: model.print_results()
    except Exception, e: print 'Error: %s' % e
    try: model.check_model()
    except Exception, e: print 'Error: %s' %e
    try: model.plot_results()
    except Exception, e: print 'Error: %s' %e

    return dataset

def classification(dataset):
    s=raw_input('Select classification technique:\n- 1. Logistic Regression\n  2. LDA\n  3. QDA\n')

    if s=='1' or s=='': # default
        model=logistic.LOGISTIC(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0]) #class labels are assumed to be in the first column

    elif s=='2':
        model=lda.LDA(dataset.data[:,1:dataset.data.shape[1]], dataset.data[:,0])

    elif s=='3':
        model=qda.QDA(dataset.data[:,1:dataset.data.shape[1]], dataset.data[:,0])

    try: model.fit_model()
    except Exception, e: print 'Error fitting model: %s' % e
    try: model.print_results()
    except Exception, e: print 'Error: %s' % e
    try: model.plot_results()
    except Exception, e: print 'Error: %s' % e

    return dataset
