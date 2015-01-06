from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import numpy as np

class LOGISTIC:
    #Implements logistic regression, with assumption checks

    def __init__(self, data, classes, penalty='l2',dual=False):
        self.data = data
        self.classes = classes
        self.penalty = penalty
        self.dual = dual
        self.classObj = linear_model.LogisticRegression(penalty=self.penalty, dual=self.dual)

    def fit_model(self):
        self.fitted_model = self.classObj.fit(self.data, self.classes)

    def checks(self):
        self.mcCheck = c.mcCheck(self.data)
        self.mcCheck.check()
        #self.acCheck = c.acCheck(self.residuals)
        #self.acCheck.check()
        #self.linCheck = c.linCheck(self.independentVar, self.dependentVar)
        #self.linCheck.check()
        self.hdCheck = c.highdimCheck(self.data)
        self.hdCheck.check()

    def Actions(self):
        self.mvnAction()
        self.eqCovAction()

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
