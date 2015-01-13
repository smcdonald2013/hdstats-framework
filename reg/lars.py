from sklearn import linear_model
import statsmodels.api as sm
import visualizations as viz
import checks as c
import regClass as rc

class LARS(rc.REG):
    """Object which performs LARS, checks assumptions, and makes plots

    Methods:
    __init__
    fit_model -- Inherits from regression base class
    CV_model
    check_model -- Inherits from regression base class
    print_results -- Inherits from regression base class
    plot_results -- Inherits from regression base class

    Instance Variables:
    self.regObj -- primary regression object, from scikit library
    self.residuals -- vector, 1xnObservations, containing residuals of fit
    """

    def __init__(self, indepVar, depVar,alpha, l1_ratio):
        rc.REG.__init__(self, indepVar, depVar)
        self.regObj = linear_model.Lars()

    def CV_model(self):
        """Perform cross-validation to select the correct alpha parameters for LARS"""
        self.regObj = linear_model.LarsCV()

    def plot_results(self):
        """Create the base regression plots as well as a qq-plot"""
        rc.REG.plot_results(self)
        viz.plot_qq(self.residuals).plot()
