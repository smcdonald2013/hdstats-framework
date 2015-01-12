from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import visualizations as viz
import numpy as np

class ELASTICNET:
    #Implements elastic-net regression, with assumption checks

    def __init__(self, indepVar, depVar,alpha=1, l1_ratio=.5):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.regObj = linear_model.ElasticNet(alpha=alpha, l1_ratio=self.l1_ratio, copy_X=False)

    def fit_model(self):
        self.fitted_model = self.regObj.fit(self.independentVar, self.dependentVar)
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def elasticnet_CV(self):
        #Chooses regularization parameter using cross-validation
        self.regObj = linear_model.ElasticNetCV()
        self.fit_model()
        self.regObj = linear_model.ElasticNet(alpha=self.regObj.alpha_, l1_ratio=self.regObj.l1_ratio_, copy_X=False)

    def check_model(self):
        #Variety of checks for elastic net fit
        self.acCheck = c.acCheck(self.residuals)
        self.acCheck.check()
        self.linCheck = c.linCheck(self.independentVar, self.dependentVar)
        self.linCheck.check()
        self.normCheck = c.normCheck(self.residuals)
        self.normCheck.check()
        self.mcCheck = c.mcCheck(self.independentVar, self.dependentVar, self.residuals)
        self.mcCheck.check()

    def print_results(self):
        print('\n Elastic Net Coefficients')
        print(self.regObj.coef_)
        print('\n R-Squared')
        print(self.regObj.score(self.independentVar,self.dependentVar))

    def plot_results(self):
        self.path = linear_model.enet_path(self.independentVar, self.dependentVar, l1_ratio=self.l1_ratio,return_models=False, fit_intercept=False)
        self.alphas = self.path[0]
        self.coefs = (self.path[1]).T
        viz.plot_regPath(self.alphas, self.coefs).plot()
