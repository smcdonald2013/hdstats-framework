from sklearn.lda import LDA as skLDA
import statsmodels.api as sm
import checks as c
import numpy as np

class LDA:
    #Implements ordinary linear discriminant analysis classification, with assumption checks

    def __init__(self, data, classes):
        self.data = data
        self.classes = classes
        self.classObj = skLDA()

    def fit_model(self):
        self.fitted_model = self.classObj.fit(self.data, self.classes)

    def checks(self):
        self.mvnCheck = c.mvnCheck(self.data)
        self.mvnCheck.check()
        self.eqCov = c.eqCovCheck(self.data, self.classes)
        self.eqCov.check()
        #self.mcCheck = c.mcCheck(self.data)
        #self.mcCheck.check()

    def Actions(self):
        self.mvnAction()
        #self.eqCovAction()

    def mvnAction(self):
        if self.mvnCheck.skewp < .05:
            print "Data does not appear to have come from a multivariate normal distribution. Logistic Regression may be more appropriate. Running now:."
            return logistic.LOGISTIC(self.data, self.classes)
        else:
            print "Data appear to be multivariate-normal."

    def eqCovAction(self):
        if self.eqCov.boxMp < .05:
            print "Classes do not appear to have similar covariance matrices. QDA would be more appropriate, performing now."
            return qda.QDA(self.data, self.classes)
        else:
            print "Classes appear to have similar covariance matrices, LDA may be justified."


