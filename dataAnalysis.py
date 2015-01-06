import sklearn # scikit-learn
import numpy as np
import ols
import sgd
import lda
import pca

def main(dataset):

    while True:
        s=raw_input('Select data analysis task:\n 1. Regression\n 2. Diminsionality reduction\n 3. Clustering\n 4. Classification\n-0. Exit\n')
        if s=='0' or s=='': # default
            break
        elif s=='1':
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
    s=raw_input('Select dimensionality reduction method:\n-1. Randomized PCA\n 2. Standard PCA\n 3. Sparse PCA\n')

    if s=='1' or s=='': # default
        s1=raw_input('Number of components to keep? (default: number of variables)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        dec=pca.RPCA(dataset.data,n_components=n_components)

    elif s=='2':
        s1=raw_input('Number of components to keep? (default: all)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        dec=pca.PCA(dataset.data,n_components=n_components)

    elif s=='3':
        s1=raw_input('Number of sparse atoms to extract? (default: number of variables)\n')
        try: n_components=int(s1)
        except: n_components=dataset.data.shape[1]

        s1=raw_input('Alpha? (Higher = more sparse. default: 1)\n')
        try: alpha=float(s1)
        except: alpha=1

        dec=pca.SPCA(dataset.data,n_components=n_components, alpha=alpha)


    try: dec.fit_model()
    except Exception, e: print 'Error fitting model: %s' % e
    try: dec.print_results()
    except Exception, e: print 'Error: %s' % e

#    dataset.dataTransformed = dec.dataTransformed

    
    return dataset

 
