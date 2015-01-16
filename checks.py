"""The checks module, which contains a variety of statistical tests.

This is the location of all of the checks/tests used in the project. 
The base check class does little beyond provide an outline that should 
be followed when implementing checks. 
"""

from sklearn import linear_model
import statsmodels.stats.outliers_influence as oi
import statsmodels.api as sm
import numpy as np
import sys, scipy, pandas
import readline
import pandas.rpy.common as com
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

class check:
    """Base check class, should be completely overwritten by child classes."""
    
    def __init__(self, data):
        """Constructor, should overwrite if more than one parameter is necessary."""
        ## Data matrix/vector, depending on the check
        self.data = data 

    def check(self):
        """Generic check function."""
        print('\n This should be overwritten!')

class mcCheck(check):
    """Tests for multicollinearity as measured by the condition number."""

    def check(self):
        """Finds condition number of data matrix."""

        ## Condition number of matrix
        self.conNum = np.linalg.cond(self.data)
        if self.conNum > 20:
            print('\nMulticollinearity is a problem')
        else:
            print('\nThere does not appear to be an issue with multicollinearity.')

class acCheck(check):
    """Tests for autocorrelation using ljung-box test.

    Generally, the input of this test will be residuals from a regression.
    """
    def check(self):
        """Performs ljung-box test, by default with 2 lags."""
        
        ##Vector with test statistics, followed by pvalues
        self.ljungbox = sm.stats.diagnostic.acorr_ljungbox(self.data, lags=2)
        if self.ljungbox[1][0] < .05:
            print('\nResiduals are autocorrelated.')
        else:
            print('\nThere does not appear to be a problem with autocorrelation of residuals.')

class linCheck(check):
    """Tests for nonlinear relationship between dependent and independent variables. 

    Note that the current implementation of this test uses the harvey-collier test, which scales poorly. 
    """
    def __init__(self, indepVar, depVar):
        """Creates a linCheck object.

        @param indepVar Array of independent variables
        @param depVar Vector of dependent variable
        """
        ## Vector of dependent variable values
        self.dependentVar = depVar
        ## Array of independent variables
        self.independentVar = indepVar

    def check(self):
        """Performs linearity check. 

        This must refit the model using the statsmodels OLS method, since
        that provides the input necessary for the harvey-collier test.
        """
        ## Statsmodel OLS object
        self.model = sm.OLS(self.dependentVar, self.independentVar[:,0])
        ## Vector giving the test statistic and p-value respectively
        self.hc = sm.stats.diagnostic.linear_harvey_collier(self.model.fit())

class normCheck(check):
    """Tests for normality of errors using anderson-darling test.""" 

    def check(self):
        """Performs anderson-darling test."""
        
        ##Vector giving the test statistic and p-value respectively
        self.norm = sm.stats.diagnostic.normal_ad(self.data)

class homoskeCheck(check):
    """Tests for heteroskedasticity."""

    def __init__(self, residuals, indepVar):
        """Constructs test.
        
        @param indepVar Array of independent variables
        @param residuals Vector of residuals
        """
        ## Array of independent variables
        self.independentVar = indepVar
        ## Vector of residuals
        self.residuals = residuals

    def check(self):
        """Performs breusch-pagan test for heteroskedasticity."""
        ##Vector containing test statistic and p-value respectively
        self.bptest = sm.stats.diagnostic.het_breushpagan(self.residuals, self.independentVar)
        if self.bptest[1] < .05:
            print('\nEvidence of heteroskedasticity.')
        else:
            print('\nHeteroskedasticity does not appear to be an issue.')

class singCheck:
    """Tests if matrix is singular."""
    
    def check(self):
        """Uses condition number of matrix to test for exact singularity."""
        if np.linalg.cond(self.data) < 1/sys.float_info.epsilon:
            ## Boolean declaring singularity status of matrix
            self.singCheck = False
            print('Data matrix is not singular.')
        else:
            print('Matrix is singular!')
            self.sing = True

class highdimCheck:
    """Tests if data is high dimensional, i.e. p>n."""

    def check(self):
        """Performs test simply by examing matrix dimensions."""
        if self.data.shape[0] < self.data.shape[1]:
            print "Data is high-dimensional!"
            ##Boolean declaring high-dimensionality status of matrix
            self.hd = True
        else:
            self.hd = False

class mvnCheck:
    """Tests if the data come from a multivariate normal distribution using R mardia test."""
    
    def check(self):
        """Performs a mardia test for multivariate normality.

        Implementation currently done using R.
        """
        importr('psych')
        pan_data = pandas.DataFrame(self.data)
        r_data = com.convert_to_r_dataframe(pan_data)
        ro.globalenv['r_data'] = r_data
        ro.r('mardia_output = mardia(r_data, plot=FALSE)')
        ro.r('pvals = c(mardia_output$p.skew, mardia_output$p.kurtosis)')
        pan_data = com.load_data('pvals')
        ##P-value for mardia skew test
        self.skewp = pan_data[0]

class eqCovCheck:
    """Tests if different classes have the same covariance matrix.

    Not currently implemented due to issues with R format.
    """
    
    def __init__(self,data,classes):
        """Class constructor.
        
        @param data data matrix
        @param classes class label vector
        """
        ##Data matrix
        self.data = data
        ##Vector of class labels
        self.classes = classes

    def check(self):
        """Performs check, some issue with output however."""
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
        ##P-value of box test of equal covariances
        self.boxMp = pan_data[0]
