from sklearn.qda import QDA as skQDA
import statsmodels.api as sm
import checks as c
import numpy as np

class QDA:
    #Implements quadratic discriminant analysis classification, with assumption checks

    def __init__(self, data, classes, classNames=False):
        self.data = data
        self.classes = classes
        self.classNames = classNames
        self.classObj = skQDA()

    def fit_model(self):
        self.fitted_model = self.classObj.fit(self.data, self.classes)

    def checks(self):
        self.mvnCheck = c.mvnCheck(self.data)
        self.mvnCheck.check()
        self.mcCheck = c.mcCheck(self.data)
        self.mcCheck.check()

    def print_results(self):
        print('\n QDA Coefficients')
        print(self.classObj.coef_)
        print('\n Class Means')
        print(self.classObj.means_)
    
    def plot_results(self):
        self.transData = self.fitted_model.transform(self.data)
        viz.plot_comps(self.transData[:,0],self.transData[:,1], compNums=[1,2],classes=self.classes, classNames=['setosa','versicolor','virginica']).plot()
