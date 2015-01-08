from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import numpy as np
import visualizations as viz

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

    def check_model(self):
        self.mcCheck = c.mcCheck(self.data)
        self.mcCheck.check()
        #self.acCheck = c.acCheck(self.residuals)
        #self.acCheck.check()
        #self.linCheck = c.linCheck(self.independentVar, self.dependentVar)
        #self.linCheck.check()

    def print_results(self):
        print('\n Logistic Coefficients')
        print(self.classObj.coef_)

    def plot_results(self):
        viz.plot_comps(self.data[:,0], self.data[:,1], compNums=[1,2], classes=self.classes, classNames=['setosa','versicolor','virginica']).plot()

    def action_model(self):
        self.mvnAction()
        self.eqCovAction()
