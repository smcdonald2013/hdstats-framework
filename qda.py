from sklearn.qda import QDA as skQDA
import statsmodels.api as sm
import checks as c
import numpy as np

class QDA:
    #Implements quadratic discriminant analysis classification, with assumption checks

    def __init__(self, data, classes):
        self.data = data
        self.classes = classes
        self.classObj = skQDA()

    def fit_model(self):
        self.fitted_model = self.classObj.fit(self.data, self.classes)

    def checks(self):
        self.mvnCheck = c.mvnCheck(self.data)
        self.mvnCheck.check()
        self.mcCheck = c.mcCheck(self.data)
        self.mcCheck.check()

    def Actions(self):
        self.mvnAction()

    def print_results(self):
        print('\n QDA Coefficients')
        print(self.classObj.coef_)
        print('\n Class Means')
        print(self.classObj.means_)
    
    def acAction(self):
        if self.acCheck.ljungbox[1][0] < .05:
            print "Residuals are autocorrelated. Implementing GLSAR."
            return sm.regression.linear_model.GLSAR(self.dependentVar, self.independentVar)
        else:
            print "Residuals appear to be uncorrelated."

    def singAction(self):
        if self.singCheck == True:
            print "Singular matrix"
            #Remove linearly dependent rows
