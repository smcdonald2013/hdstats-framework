"""@package docstring
The Lasso regularized regression module."""
from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import visualizations as viz
import numpy as np
import regClass as rc

class LASSO(rc.REG):
    """Object which performs lasso regression, checks assumptions, and makes plots."""

    def __init__(self, indepVar, depVar,alpha=1):
        """Lasso constructor

        @param indepVar Array of independent variables
        @param depVar Vector of dependent variable
        @param alpha The regularization parameter
        """
        rc.REG.__init__(self, indepVar, depVar)
        ##Regularization parameter
        self.alpha = alpha
        ##Lasso object, from Scikit-Learn
        self.regObj = linear_model.Lasso(alpha=alpha, fit_intercept=False, copy_X=False)

    def fit_model(self):
        """Fit the lasso model and add sparsity variable"""
        rc.REG.fit_model(self)
        ##Fraction of nonzero coefficients
        self.sparsity = float(np.count_nonzero(self.regObj.coef_))/self.regObj.coef_.shape[0]

    def CV_model(self):
        """Perform cross-validation to select the correct alpha parameter for the lasso regression"""
        self.regObj = linear_model.LassoCV()
        self.fit_model()
        self.regObj = linear_model.Lasso(alpha=self.regObj.alpha_, copy_X=False)

    def lasso_lars_CV(self):
        """Performs cross-validation using LARS algorithm. 

        This method should be used when multicollinearity isn't a problem, and p >> n.
        """
        self.regObj = linear_model.LassoLarsCV()
        self.fit_model()
        self.regObj = linear_model.Lasso(alpha=self.regObj.alpha_, copy_X=False)

    def print_results(self):
        """Prints base results and sparsity."""
        rc.REG.print_results(self)
        print('\n Sparsity of the solution is: ')
        print('%.3f' % (self.sparsity))

    def plot_results(self):
        """Create the base regression plots as well as a regularization path plot."""
        rc.REG.plot_results(self)
        path = linear_model.lasso_path(self.independentVar, self.dependentVar, return_models=False, fit_intercept=False)
        alphas = path[0] #Vector of alphas
        coefs = (path[1]).T #Array of coefficients for each alpha
        viz.plot_regPath(alphas, coefs).plot()
