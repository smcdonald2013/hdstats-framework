from sklearn import linear_model
import statsmodels.stats.outliers_influence as oi
import statsmodels.api as sm
import numpy as np
import sys

class mcCheck:
    #Tests for multicollinearity in the design matrix by performing VIF test on each dependent variable

    def __init__(self, indepVar, depVar, residuals):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.residuals = residuals

    def check(self):
        self.conNum = np.linalg.cond(self.independentVar)
        self.nVars = self.independentVar.shape[1]
        self.vif = np.empty(self.nVars)
        for i in range(self.nVars-1):
            self.vif[i] = oi.variance_inflation_factor(self.independentVar,i)

class acCheck:
    #Tests for autocorrelation of the residuals using the durbin-watson test

    def __init__(self, residuals):
        self.residuals = residuals

    def check(self):
        #self.dw = sm.stats.stattools.durbin_watson(self.residuals)
        self.ljungbox = sm.stats.diagnostic.acorr_ljungbox(self.residuals, lags=2)
        self.acf = sm.tsa.stattools.acf(self.residuals)
        #PACF is behaving strangely
        #self.pacf = sm.tsa.stattools.pacf(self.residuals)

class linCheck:
    #Tests for nonlinear relationship between x's and y's. 

    def __init__(self, indepVar, depVar):
        self.dependentVar = depVar
        self.independentVar = indepVar

    def check(self):
        self.model = sm.OLS(self.dependentVar, self.independentVar[:,0])
        self.hc = sm.stats.diagnostic.linear_harvey_collier(self.model.fit())
        #self.rain = sm.stats.diagnostic.linear_rainbow(self.model.fit())

class normCheck:
    #Tests for normality of errors. 

    def __init__(self, residuals):
        self.residuals = residuals

    def check(self):
        self.norm = sm.stats.diagnostic.normal_ad(self.residuals)

class homoskeCheck:
    #Tests for heteroskedasticity

    def __init__(self, residuals, indepVar):
        self.independentVar = indepVar
        self.residuals = residuals

    def check(self):
        self.bptest = sm.stats.diagnostic.het_breushpagan(self.residuals, self.independentVar)

class singCheck:
    #Tests if matrix is singular, because for some unfathomable reason scikit does not throw an exception when running a regression with singular design matrix

    def __init__(self,indepVar):
        self.independentVar = indepVar

    def check(self):
        if np.linalg.cond(self.independentVar) < 1/sys.float_info.epsilon:
            #Matrix is good!
            self.singCheck = False
        else:
            print "Matrix is singular!"
            self.sing = True

class highdimCheck:
    #Tests if data is high dimensional, i.e. p>n
    
    def __init__(self,indepVar):
        self.independentVar = indepVar

    def check(self):
        if self.independentVar.shape[0] < self.independentVar.shape[1]:
            print "Data is high-dimensional!"
            self.hd = True
        else:
            self.hd = False
