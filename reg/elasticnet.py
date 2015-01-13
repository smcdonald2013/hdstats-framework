from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import visualizations as viz
import numpy as np
import regClass as rc

class ELASTICNET(rc.REG):
    """Object which performs elastic-net regression, checks assumptions, and makes plots

    Methods:
    __init__
    fit_model
    CV_model
    check_model -- Inherits from regression base class
    print_results
    plot_results

    Instance Variables:
    self.regObj -- primary regression object, from scikit library
    self.residuals -- vector, 1xnObservations, containing residuals of fit
    """

    def __init__(self, indepVar, depVar,alpha=1, l1_ratio=.5):
        """Constructs class object, including variable setting and elastic-net object creation

        Instance Variables:
        self.independentVar -- Inherited from regression baseclass
        self.dependentVar -- Inherited from regression baseclass
        self.alpha
        self.regObj
        """
        rc.REG.__init(self, indepVar, depVar)
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.regObj = linear_model.ElasticNet(alpha=alpha, l1_ratio=self.l1_ratio, copy_X=False)

    def fit_model(self):
        """Fit the elastic-net model. Inherits from the regression baseclass, and adds sparsity variable"""
        rc.REG.fit_model(self)
        self.sparsity = float(np.count_nonzero(self.regObj.coef_))/self.regObj.coef_.shape[0]

    def CV_model(self):
        """Perform cross-validation to select the correct alpha and l1_ratio parameters for the elastic net regression"""
        self.regObj = linear_model.ElasticNetCV()
        self.fit_model()
        self.regObj = linear_model.ElasticNet(alpha=self.regObj.alpha_, l1_ratio=self.regObj.l1_ratio_, copy_X=False)

    def print_results(self):
        """Prints useful information for a elastic-net regression.

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
        self.path = linear_model.enet_path(self.independentVar, self.dependentVar, l1_ratio=self.l1_ratio,return_models=False, fit_intercept=False)
        self.alphas = self.path[0]
        self.coefs = (self.path[1]).T
        viz.plot_regPath(self.alphas, self.coefs).plot()
