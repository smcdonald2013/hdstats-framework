from sklearn import linear_model
import statsmodels.stats.outliers_influence as oi
import statsmodels.api as sm
import numpy as np
import sys
import scipy
import pandas

import readline
import pandas.rpy.common as com
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

class mcCheck:
    #Tests for multicollinearity in the design matrix

    def __init__(self, indepVar, depVar=0, residuals=0):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.residuals = residuals

    def check(self):
        self.conNum = np.linalg.cond(self.independentVar)
        if self.dependentVar != 0:
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

class mvnCheck:
    #Tests if the data come from a multivariate normal distribution
    def __init__(self,data):
        self.data = data

    def check(self):
        importr('psych')
        pan_data = pandas.DataFrame(self.data)
        r_data = com.convert_to_r_dataframe(pan_data)
        ro.globalenv['r_data'] = r_data
        ro.r('mardia_output = mardia(r_data, plot=FALSE)')
        ro.r('pvals = c(mardia_output$p.skew, mardia_output$p.kurtosis)')
        pan_data = com.load_data('pvals')
        self.skewp = pan_data[0]
        #self.kurp = pan_data[1] R mardia function does not currently implement kurtosis test

class eqCovCheck:
    #Tests if different classes have the same covariance matrix
    
    def __init__(self,data,classes):
        self.data = data
        self.classes = classes

    def check(self):
        importr('biotools')
        pan_data = pandas.DataFrame(self.data)
        pan_classes = pandas.DataFrame(self.classes)
        r_data = com.convert_to_r_dataframe(pan_data)
        r_classes = com.convert_to_r_dataframe(pan_classes)
        ro.globalenv['r_data'] = r_data
        ro.globalenv['r_classes'] = r_classes
        ro.r('boxM_test = boxM(data=r_data,grouping=r_classes)')
        ro.r('pvals = boxM_test$p.value')
        pan_data = com.load_data('pvals')
        self.boxMp = pan_data[0]

class conIndCheck:
    #Tests if the data variables are independent, conditional on the class labels. Note that since this test is only reliable if the underlying data are gaussian, it is really a correlation test

    def __init__(self,data,classes):
        self.data = data
        self.classes = classes
        self.bnlearn = importr('bnlearn')
