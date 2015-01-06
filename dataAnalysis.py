import sklearn # scikit-learn
import numpy as np
import ols
import sgd
import lda

def main(dataset):

    while True:
        s=raw_input('Select data analysis task:\n 1. Print data to screen\n 2. Regression\n 3. Classification\n 0. Exit\n')
        if s=='0':
            break
        elif s=='1':
            np.set_printoptions(precision=3, suppress=True)
            print(dataset.data)
            #return dataset
        elif s=='2':
            if dataset.data.shape[0] < 100000:
                print('Running OLS Regression')
                reg = ols.OLS(dataset.data[:,1:dataset.data.shape[1]],dataset.data[:,0])
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




