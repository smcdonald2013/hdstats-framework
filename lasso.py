from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import visualizations as viz
import numpy as np

class LASSO:
    #Implements lasso regression, with assumption checks

    def __init__(self, indepVar, depVar,alpha=1):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.Lasso(alpha=alpha, fit_intercept=False, copy_X=False)

    def fit_model(self):
        self.fitted_model = self.regObj.fit(self.independentVar, self.dependentVar)
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def lasso_CV(self):
        #Performs cross-validation to select the optimal alpha for lasso regression
        self.regObj = linear_model.LassoCV()
        self.fit_model()
        self.regObj = linear_model.Lasso(alpha=self.regObj.alpha_, copy_X=False)

    def lasso_lars_CV(self):
        #Cross-validation using LARS algorithm. Should be used when multicollinearity isn't a problem, and p >> n
        self.regObj = linear_model.LassoLarsCV()
        self.fit_model()
        self.regObj = linear_model.Lasso(alpha=self.regObj.alpha_, copy_X=False)

    def checks(self):
        #Variety of checks for lasso fit
        self.acCheck = c.acCheck(self.residuals)
        self.acCheck.check()
        self.linCheck = c.linCheck(self.independentVar, self.dependentVar)
        self.linCheck.check()
        self.normCheck = c.normCheck(self.residuals)
        self.normCheck.check()
        self.mcCheck = c.mcCheck(self.independentVar, self.dependentVar, self.residuals)
        self.mcCheck.check()

    def print_results(self):
        print('\n Lasso Coefficients')
        print(self.regObj.coef_)
        print('\n R-Squared')
        print(self.regObj.score(self.independentVar,self.dependentVar))

    def plot_results(self):
        self.path = linear_model.lasso_path(self.independentVar, self.dependentVar, return_models=False, fit_intercept=False)
        self.alphas = self.path[0]
        self.coefs = (self.path[1]).T
        viz.plot_regPath(self.alphas, self.coefs).plot()
