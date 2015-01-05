import sklearn # scikit-learn
import numpy as np
import ols
import sgd
import pca

def main(dataset):

    while True:
        s=raw_input('Select data analysis task:\n 1. Print data to screen\n 2. Regression\n 3. Diminsionality reduction\n 0. Exit\n')
        if s=='0':
            break
        elif s=='1':
            np.set_printoptions(precision=3, suppress=True)
            print(dataset.data)
            #return dataset
        elif s=='2':
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

        elif s=='3':
            print('Trying randomized PCA\n')
            dec=pca.RPCA(dataset.data)
            dec.fit_model()
            print('Components')
            print(dec.obj.components_)
            print('\n Explained Variance Ratio')
            print(dec.obj.explained_variance_ratio_)

        else:
            print('Input not recognized')
            return dataset




