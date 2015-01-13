from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import visualizations as viz

class RIDGE:
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

    def __init__(self, indepVar, depVar,alpha=1):
        """Constructs class object, including variable setting and elastic-net object creation

        Instance Variables:
        self.independentVar -- Inherited from regression baseclass
        self.dependentVar -- Inherited from regression baseclass
        self.alpha
        self.regObj
        """
        rc.REG.__init__(self, indepVar, depVar)
        self.alpha = alpha
        self.regObj = linear_model.Ridge(alpha=alpha, copy_X=False)

    def CV_model(self):
        """Perform cross-validation to select the correct alpha and l1_ratio parameters for the ridge regression"""
        self.regObj = linear_model.RidgeCV()
        self.fit_model()
        self.regObj = linear_model.Ridge(alpha=self.regObj.alpha_, copy_X=False)
