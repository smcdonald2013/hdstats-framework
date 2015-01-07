import sklearn # scikit-learn
import numpy as np
import ols
import sgd
import lda
import pca

def main(dataset):

    while True:
        s=raw_input('Select data analysis task:\n 1. Regression\n 2. Diminsionality reduction\n 3. Clustering\n 4. Classification\n 0. Exit\n')
        if s=='0' or s=='': # default
            break
        elif s=='1':
            dataset=regression(dataset)
        elif s=='10':
            if dataset.data.shape[0] < 100000:
                print('Running OLS Regression')
                reg = ols.OLS(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0]) #independent variable is assumed to be in the first column
                reg.fit_model()
                print('Regression summary', reg.fitted_model.coef_)
                print('Performing assumption checks')
                reg.checks()
                print('Taking any necessary corrective actions')
                reg.mcAction()
                reg.acAction()
                reg.linAction()
                reg.singAction()
                reg.homoskeAction()
            else:
                print('Dataset is very large, running SGD Regression')
                reg = sgd.SGD(dataset.data[:,1:dataset.data.shape[1]], dataset.data[:,0])
                reg.fit_model()
                print('Regression summary', reg.fitted_model.coef_)

        elif s=='2':
            dataset=dimensionalityReduction(dataset)

        elif s=='3':
            s=raw_input('Number of categories? Enter zero if unknown\n')
            try: n=int(s)
            except: n=0
            
            if n>0 & dataset.data.shape[0] < 10000:
                print('Known number of categories and <10k samples: Using KMeans clustering\n')
                # Use KMeans clustering
                # Followed by Spectral Clustering or GMM in event of failure
            elif n>0:
                print('Known number of categories and >10k samples: Using MiniBatch KMeans\n')
                # Use MiniBatch KMeans
            elif dataset.data.shape[0] < 10000:
                print('Unknown number of categories and <10k samples: Using MeanShift')
                # Use MeanShift or VBGMM
            else:
                print('Too many samples to analyze without knowing number of categories\n')

        elif s=='4':
            if dataset.data.shape[0] < 100000:
                print('Running LDA')
                classifier = lda.LDA(dataset.data[:,1:dataset.data.shape[1]], dataset.data[:,0])
                classifier.fit_model()
                print('Classification summary', classifier.fitted_model.coef_)
                print('Performing assumption checks')
                classifier.checks()
                print('Taking any necessary corrective action')
                classifier.mvnAction()
        else:
            print('Input not recognized')

    return dataset



def dimensionalityReduction(dataset):
    s=raw_input('Select dimensionality reduction method:\n 1. Principal Component Analysis (PCA)\n-2. Randomized PCA (faster)\n 3. Sparse PCA (finds sparse principal components)\n 4. Independent Component Analysis (ICA - components need not be orthogonal)\n 5. Isometric Mapping (Isomap)\n 6. Locally Linear Embedding\n 7. Spectral Embedding\n 8. Guide me\n')

    if s=='1':
        s1=raw_input('Number of components to keep? (default: all)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        dec=pca.PCA(dataset.data,n_components=n_components)

    elif s=='2' or s=='': # default
        s1=raw_input('Number of components to keep? (default: all)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        dec=pca.RPCA(dataset.data,n_components=n_components)

    elif s=='3':
        s1=raw_input('Number of sparse atoms to extract? (default: number of variables)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        s1=raw_input('Alpha? (Higher = more sparse. default: 1)\n')
        try: alpha=float(s1)
        except: alpha=1

        dec=pca.SPCA(dataset.data,n_components=n_components, alpha=alpha)

    elif s=='4':
        s1=raw_input('Number of components to use? (default: number of variables)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        dec=pca.ICA(dataset.data,n_components=n_components)

    elif s=='5':
        s1=raw_input('Number of components to use? (default: 2)\n')
        try: n_components=int(s1)
        except: n_components=2

        s1=raw_input('Number of neighbors to consider for each point.? (default: 5)\n')
        try: n_neighbors=int(s1)
        except: n_neighbors=5

        dec=pca.Isomap(dataset.data, n_components=n_components, n_neighbors=n_neighbors)

    elif s=='6':
        s1=raw_input('Number of components to use? (default: 2)\n')
        try: n_components=int(s1)
        except: n_components=2

        s1=raw_input('Number of neighbors to consider for each point.? (default: 5)\n')
        try: n_neighbors=int(s1)
        except: n_neighbors=5

        dec=pca.LocallyLinearEmbedding(dataset.data, n_components=n_components, n_neighbors=n_neighbors)
    elif s=='7':
        s1=raw_input('Number of components to use? (default: 2)\n')
        try: n_components=int(s1)
        except: n_components=2

        s1=raw_input('Number of neighbors to consider for each point.? (default: 5)\n')
        try: n_neighbors=int(s1)
        except: n_neighbors=5

        dec=pca.SpectralEmbedding(dataset.data, n_components=n_components, n_neighbors=n_neighbors)

    elif s=='8':
        s1=raw_input('Do you want to extract components that are orthogonal to each other?\n 0. No\n-1. Yes\n')
        s2=raw_input('How many components do you want to find? (default: number of variables)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        if s1=='0':
            print('Trying ICA')
            dec=pca.ICA(dataset.data,n_components=n_components)
        else:
            print('Trying Randomized PCA')
            dec=pca.RPCA(dataset.data,n_components=n_components)


    try: dec.fit_model()
    except Exception, e: print 'Error fitting model: %s' % e
    try: dec.print_results()
    except Exception, e: print 'Error: %s' % e

#    dataset.dataTransformed = dec.dataTransformed
    
    return dataset
 
def regression(dataset):
    sp = raw_input('Is the underlying model presumed to be sparse? (Default is no)\n 1.Yes\n 2.No\n')
    if sp=='1':
        spVal = True
    else:
        spVal = False
    s=raw_input('Select regression technique:\n 1. OLS\n 2. Lasso\n 3. Ridge\n 4. Elastic Net\n 5. Lars\n 6. OMP\n 0. Guide me\n')

    if s=='1' or s=='': # default
        reg=ols.OLS(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], sparse=sp) #independent variable is assumed to be in the first column

    elif s=='2':
        s1=raw_input('Value of alpha parameter? (default: 1)\n')
        try: alpha=int(s1)
        except: alpha=1

        reg=lasso.LASSO(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], alpha=alpha)

    elif s=='3':
        s1=raw_input('Value of alpha parameter? (default: 1)\n')
        try: alpha=int(s1)
        except: alpha=1

        reg=ridge.RIDGE(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], alpha=alpha)

    elif s=='4':
        s1=raw_input('Value of alpha parameter? (default: 1)\n')
        try: alpha=int(s1)
        except: alpha=1

        s1=raw_input('Ratio of l1 penalty relative to l2? (default: .5)\n')
        try: l1_ratio = int(s1)
        except: l1_ratio=.5

        reg=elasticnet.ELASTICNET(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], alpha=alpha, l1_ratio=l1_ratio)

    elif s=='5':
        s1=raw_input('Value of alpha parameter? (default: 1)\n')
        try: alpha=int(s1)
        except: alpha=1

        s1=raw_input('Ratio of l1 penalty relative to l2? (default: .5)\n')
        try: l1_ratio = int(s1)
        except: l1_ratio=.5

        reg=lars.LARS(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0], alpha=alpha, l1_ratio=l1_ratio)

    elif s=='6':

        reg=omp.OMP(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0])

    try: reg.fit_model()
    except Exception, e: print 'Error fitting model: %s' % e
    #try: dec.print_results()
    #except Exception, e: print 'Error: %s' % e

#    dataset.dataTransformed = dec.dataTransformed

    return dataset
