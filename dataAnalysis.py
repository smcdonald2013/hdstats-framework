import sklearn # scikit-learn
import numpy as np
import dimensionalityReduction
import clustering
from reg import ols, sgd, lasso, elasticnet, ridge, lars, omp, glsar
from reg import boxcox as bc
from classification import lda, qda, logistic

def main(dataset):
    while True:
        s=raw_input('Select data analysis task:\n  1. Regression\n  2. Dimensionality reduction\n  3. Clustering\n  4. Classification\n- 0. Exit\n')
        if s=='0' or s=='': # default
            break
        elif s=='1':
            dataset=regression(dataset)
        elif s=='2':
            dataset=dimensionalityReductionAnalysis(dataset)
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


def dimensionalityReductionAnalysis(dataset):
    s=raw_input('Select dimensionality reduction method:\n  1. Principal Component Analysis (PCA)\n- 2. Randomized PCA (faster)\n  3. Sparse PCA (finds sparse principal components)\n  4. Independent Component Analysis (ICA - components need not be orthogonal)\n  5. Isometric Mapping (Isomap)\n  6. Locally Linear Embedding\n  7. Spectral Embedding\n  8. Guide me\n')

    if s=='1':
        s1=raw_input('Number of components to keep? (default: all)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        model=dimensionalityReduction.PCA(dataset.data,n_components=n_components)

    elif s=='2' or s=='': # default
        s1=raw_input('Number of components to keep? (default: all)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        model=dimensionalityReduction.RPCA(dataset.data,n_components=n_components)

    elif s=='3':
        s1=raw_input('Number of sparse atoms to extract? (default: number of variables)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        s1=raw_input('Alpha? (Higher = more sparse. default: 1)\n')
        try: alpha=float(s1)
        except: alpha=1

        model=dimensionalityReduction.SPCA(dataset.data,n_components=n_components, alpha=alpha)

    elif s=='4':
        s1=raw_input('Number of components to use? (default: number of variables)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        model=dimensionalityReduction.ICA(dataset.data,n_components=n_components)

    elif s=='5':
        s1=raw_input('Number of components to use? (default: 2)\n')
        try: n_components=int(s1)
        except: n_components=2

        s1=raw_input('Number of neighbors to consider for each point.? (default: 5)\n')
        try: n_neighbors=int(s1)
        except: n_neighbors=5

        model=dimensionalityReduction.Isomap(dataset.data, n_components=n_components, n_neighbors=n_neighbors)

    elif s=='6':
        s1=raw_input('Number of components to use? (default: 2)\n')
        try: n_components=int(s1)
        except: n_components=2

        s1=raw_input('Number of neighbors to consider for each point.? (default: 5)\n')
        try: n_neighbors=int(s1)
        except: n_neighbors=5

        model=dimensionalityReduction.LocallyLinearEmbedding(dataset.data, n_components=n_components, n_neighbors=n_neighbors)
    elif s=='7':
        s1=raw_input('Number of components to use? (default: 2)\n')
        try: n_components=int(s1)
        except: n_components=2

        s1=raw_input('Number of neighbors to consider for each point.? (default: 5)\n')
        try: n_neighbors=int(s1)
        except: n_neighbors=5

        model=dimensionalityReduction.SpectralEmbedding(dataset.data, n_components=n_components, n_neighbors=n_neighbors)

    elif s=='8':
        s1=raw_input('Do you want to extract components that are orthogonal to each other?\n 0. No\n-1. Yes\n')
        s2=raw_input('How many components do you want to find? (default: number of variables)\n')
        try: n_components=int(s2)
        except: n_components=dataset.data.shape[1]

        if s1=='0':
            print 'Trying ICA'
            model=dimensionalityReduction.ICA(dataset.data,n_components=n_components)
        else:
            print 'Trying Randomized PCA'
            model=dimensionalityReduction.RPCA(dataset.data,n_components=n_components)


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
    s=raw_input('Select regression technique:\n- 1. OLS\n  2. Lasso\n  3. Ridge\n  4. Elastic Net\n  5. Lars\n  6. OMP\n  7. Guide me\n')

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

    elif s=='7':
        sp = raw_input('Is the underlying model assumed to be sparse? (Default is no)\n  1. Yes\n- 0. No\n')
        if sp=='1':
            spVal = True
        else:
            spVal = False
        print('\nRunning OLS Regression')
        model = ols.OLS(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], sparse=spVal) #independent variable is assumed to be in the first column
        model.fit_model()
        print('\nRegression summary')
        model.print_results()
        print('\nNecessary plots')
        model.plot_results()
        print('\nPerforming assumption checks')
        model.check_model()
        print('\nTaking any necessary corrective actions')
        model = regGuide(model)

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

def regGuide(model):
    """Call the action functions, which consider the output of the checks and fit additional models as required. The order in which the action functions are called is important, as it is designed to fix more problematic assumption fails first. Return the adjusted model."""
    #NewModel = mcAction(model)
    NewModel = acAction(model)
    #NewModel = linAction(model)
    #NewModel = singAction(model)
    #NewModel = homoskeAction(model)
    return NewModel

def mcAction(model):
    """This function takes the appropriate action given the presence of multicollinearity and the sparsity of the data.

    There are 4 possible situations:
    Multicollinearity exists, and the data are sparse-- ElasticNet
    Multicollinearity exists, the data are not sparse-- Ridge
    Multicollinearity does not exist, but the data are sparse--Lasso
    Multicollinearity does not exist, and the data aren't sparse--keep OLS
    """
    if model.mcCheck.conNum > 20:
        print('\nMulticollinearity is a problem. Condition number of design matrix is ' , self.mcCheck.conNum)
        if model.sparse == True:
            print('\nThe underlying model is also sparse. Fitting elastic-net regression.')
            return elasticnet.ELASTICNET(model.independentVar, model.dependentVar, alpha=.5, l1_ratio=.5)
        else:
            print('\nThe underlying model is not sparse. Fitting ridge regression.')
            return ridge.RIDGE(model.independentVar, model.dependentVar, alpha=1)
    elif model.highdimCheck == True:
        print('\nMulticollinearity is not an issue, but the data is high dimensional. Fitting lasso regression.')
        return lasso.LASSO(model.independentVar, model.dependentVar)
    else:
        return model

def acAction(model):
    """This method takes a model and runs a GLSAR if the residuals are autocorrelated.

       Note that Cochrane-orcutt would be the traditional procedure here, however Statsmodels does not implement it. GLSAR appears to be similar"""
    if model.acCheck.ljungbox[1][0] < .05:
        print('\nResiduals are autocorrelated. Implementing GLSAR.')
        return glsar.GLSAR(model.dependentVar, model.independentVar)
    else:
        print('\nResiduals appear to be uncorrelated.')
        return model

def linAction(model):
    """If the model failed the linearity check, this method transforms the variables using a box-cox transformation.

    Note that the method currently uses the correlation between the independent and dependent variables, but a preferred method would use the residuals and independent variables instead.
    """
    if model.linCheck.hc[1] < .05:
        print('Linear model is incorrect, transforming variables using box-cox transformation')
        trans = np.empty([model.independentVar.shape[0],model.independentVar.shape[1]])
        for i in range(model.independentVar.shape[1]):
            linData = bc.LINTRANS(model.independentVar[:,i], model.dependentVar)
            """In the future, this should probably be redone using the correlation between the residuals and the independent variables."""
            linData.linearize()
            trans[:,i] = linData.opTrans
        return  ols.OLS(trans, model.dependentVar)
    else:
        print('\nLinear model appears reasonable.')
        return model

def singAction(model):
    """If the data matrix is singular, this throws a warning, but no automatic procedure is implemented to make the matrix singular, as it is presumed the user will want to inspect the matrix to determine the best approach."""
    if model.singCheck == True:
        print('\nSingular data matrix. Inspect data and remove linearly dependent samples.')
    return model

def homoskeAction(model):
    """If there is evidence of heteroskedasticity, there are procedures that can address it, however they tend to require knowledge about the underlying process that is rarely satisfied in practice. Instead, we recommend using robust standard errors, as implemented in this function."""  
    if model.homoskeCheck.bptest[1] < .05:
        print('\nEvidence of heteroskedasticity. Use only robust standard errors.')
    else:
        print('\nHeteroskedasticity does not appear to be a problem.')

    return model
