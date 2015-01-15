"""@package docstring
The ridge module, which performs l2 regularization."""
from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import visualizations as viz
import regClass as rc

class RIDGE(rc.REG):
    """Object which performs ridge regression, checks assumptions, and makes plots."""

    def __init__(self, indepVar, depVar,alpha=1):
        """Ridge constructor

        @param indepVar Array of independent variables
        @param depVar Vector of dependent variables
        """
        rc.REG.__init__(self, indepVar, depVar)
        ## Regularization parameters
        self.alpha = alpha
        ## Ridge Model Object (from Scikit-learn)
        self.regObj = linear_model.Ridge(alpha=alpha, copy_X=False)

    def CV_model(self):
        """Perform cross-validation to select the correct alpha and l1_ratio parameters for the ridge regression."""
        self.regObj = linear_model.RidgeCV()
        self.fit_model()
        self.regObj = linear_model.Ridge(alpha=self.regObj.alpha_, copy_X=False)
