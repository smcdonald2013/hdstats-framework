from sklearn import linear_model
import statsmodels.stats.outliers_influence as oi
import statsmodels.api as sm
import numpy as np

class mcCheck:
    #Tests for multicollinearity in the design matrix by performing VIF test on each dependent variable

    def __init__(self, indepVar, depVar, residuals):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.residuals = residuals

    def check(self):
        self.nVars = self.dependentVar.shape[1]
        self.vif = np.empty(self.nVars)
        for i in range(self.nVars-1):
            self.vif[i] = oi.variance_inflation_factor(self.dependentVar,i)

class acCheck:
    #Tests for autocorrelation of the residuals using the durbin-watson test

    def __init__(self, residuals):
        self.residuals = residuals

    def check(self):
        self.dw = sm.stats.stattools.durbin_watson(self.residuals)
        self.acf = sm.stats.stattools.acf(self.residuals)
        self.pacf = sm.stats.stattools.pacf(self.residuals)

class linCheck:
    #Tests for nonlinear relationship between x's and y's. 

    def __init__(self, indepVar, depVar, residuals):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.residuals = residuals

    def check(self):
        self.model = sm.OLS(self.dependentVar, self.independentVar)
        self.hc = sm.stats.diagnostic.linear_harvey_collier(self.model.fit())

class normCheck:
    #Tests for normality of errors. 

    def __init__(self, residuals):
        self.residuals = residuals

    def check(self):
        self.hc = sm.stats.diagnostic.normal_ad(self.residuals)
