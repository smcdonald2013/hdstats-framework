from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import visualizations as viz
import numpy as np
import regClass as rc

class LASSO(rc.REG):
    """Object which performs lasso regression, checks assumptions, and makes plots

    Methods:
    __init__
    fit_model
    CV_model
    lasso_lars_CV
    check_model -- Inherits from regression base class
    print_results
    plot_results
    model_actions

    Instance Variables:
    self.regObj -- primary regression object, from scikit library
    self.residuals -- vector, 1xnObservations, containing residuals of fit
    """

    def __init__(self, indepVar, depVar,alpha=1):
        """Constructs class object, including variable setting and lasso object creation

        Instance Variables:
        self.independentVar -- Inherited from regression baseclass
        self.dependentVar -- Inherited from regression baseclass
        self.alpha
        self.regObj
        """
        rc.REG.__init__(self, indepVar, depVar)
        self.alpha = alpha
        self.regObj = linear_model.Lasso(alpha=alpha, fit_intercept=False, copy_X=False)

    def fit_model(self):
        """Fit the lasso model. Inherits from the regression baseclass, and adds sparsity variable"""
        rc.REG.fit_model(self)
        self.sparsity = float(np.count_nonzero(self.regObj.coef_))/self.regObj.coef_.shape[0]

    def CV_model(self):
        """Perform cross-validation to select the correct alpha parameter for the lasso regression"""
        self.regObj = linear_model.LassoCV()
        self.fit_model()
        self.regObj = linear_model.Lasso(alpha=self.regObj.alpha_, copy_X=False)

    def lasso_lars_CV(self):
        """Performs cross-validation using LARS algorithm. 

        This method should be used when multicollinearity isn't a problem, and p >> n
        """
        self.regObj = linear_model.LassoLarsCV()
        self.fit_model()
        self.regObj = linear_model.Lasso(alpha=self.regObj.alpha_, copy_X=False)

    def print_results(self):
        """Prints useful information for a lasso regression.

        Prints:
        Coefficients -- Inherited from regression base class
        R-Squared -- Inherited from regression base class
        Sparsity -- The fraction of nonzero coefficients
        """
        rc.REG.print_results(self)
        print('\n Sparsity of the solution is: ')
        print('%.3f' % (self.sparsity))

    def plot_results(self):
        """Create the base regression plots as well as a regularization path plot
        """
        rc.REG.plot_results(self)
        self.path = linear_model.lasso_path(self.independentVar, self.dependentVar, return_models=False, fit_intercept=False)
        self.alphas = self.path[0]
        self.coefs = (self.path[1]).T
        viz.plot_regPath(self.alphas, self.coefs).plot()
