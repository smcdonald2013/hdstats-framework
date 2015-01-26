"""@package docstring
The elastic net module, which combines l1 and l2 penalties."""
from sklearn import linear_model
import checks as c
import visualizations as viz
import regClass as rc

class ELASTICNET(rc.REG):
    """Object which performs elastic-net regression, checks assumptions, and makes plots."""

    def __init__(self, indepVar, depVar,alpha=1, l1_ratio=.5):
        """Elastic net constructor

        @param indepVar Array of independent variables
        @param depVar Vector of dependent variable
        @param alpha Regularization parameter (Defaults to 1)
        @param l1_ratio Ratio of l1 to l2 penalties
        """
        rc.REG.__init__(self, indepVar, depVar)
        ## Ratio of l1 to l2 penalties
        self.l1_ratio = l1_ratio
        ## Regularization parameter
        self.alpha = alpha
        ## Elastic-Net Object (from Scikit-learn)
        self.regObj = linear_model.ElasticNet(alpha=alpha, l1_ratio=self.l1_ratio, copy_X=False)

    def fit_model(self):
        """Fit the elastic-net model, add sparsity member variable."""
        rc.REG.fit_model(self)
        ## Fraction of nonzero coefficients
        self.sparsity = float(np.count_nonzero(self.regObj.coef_))/self.regObj.coef_.shape[0]

    def CV_model(self):
        """Perform cross-validation to select the correct alpha and l1_ratio parameters for the elastic net regression"""
        self.regObj = linear_model.ElasticNetCV()
        self.fit_model()
        self.regObj = linear_model.ElasticNet(alpha=self.regObj.alpha_, l1_ratio=self.regObj.l1_ratio_, copy_X=False)

    def print_results(self):
        """Prints useful information for a elastic-net regression."""
        rc.REG.print_results(self)
        print('\n Sparsity of the solution is: ')
        print('%.3f' % (self.sparsity))

    def plot_results(self):
        """Create the base regression plots as well as a regularization path plot."""
        rc.REG.plot_results(self)
        path = linear_model.enet_path(self.independentVar, self.dependentVar, l1_ratio=self.l1_ratio,return_models=False, fit_intercept=False)
        alphas = path[0] #Vector of alphas
        coefs = (path[1]).T #Array of coefficients for each alpha
        viz.plot_regPath(alphas, coefs).plot()
