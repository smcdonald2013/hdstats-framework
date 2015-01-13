from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import visualizations as viz
import regClass as rc

class OMP(rc.REG):
    """Object which performs orthogonal matching pursuit regression, checks assumptions, and makes plots

    Methods:
    __init__
    fit_model -- Inherits from regression base class
    check_model -- Inherits from regression base class
    print_results -- Inherits from regression base class
    plot_results

    Instance Variables:
    self.regObj -- primary regression object, from scikit library
    self.residuals -- vector, 1xnObservations, containing residuals of fit
    """

    def __init__(self, indepVar, depVar):
        """Constructs class object, including variable setting and lasso object creation

        Instance Variables:
        self.independentVar -- Inherited from regression baseclass
        self.dependentVar -- Inherited from regression baseclass
        self.alpha
        self.regObj
        """
        rc.REG.__init__(self, indepVar, depVar)
        self.regObj = linear_model.OrthogonalMatchingPursuit()

    def plot_results(self):
        """Create the base regression plots as well as a regularization path plot
        """
        rc.REG.plot_results(self)
        viz.plot_qq(self.residuals).plot()
