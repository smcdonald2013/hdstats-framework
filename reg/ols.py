from sklearn import linear_model
import checks as c
import visualizations as viz
import regClass as rc

class OLS(rc.REG):
    """Object which performs ordinary least squares regression, checks assumptions, and makes plots

    Methods:
    __init__ -- inherits from regression base class
    fit_model -- inherits from regression base class
    check_model
    print_results -- inherits from regression base class
    plot_results

    Instance Variables:
    self.regObj -- primary regression object, from scikit library
    self.residuals -- vector, 1xnObservations, containing residuals of fit
    """

    def check_model(self):
        """Checks assumptions of OLS regression. Inherits 4 from regression base class and adds 3 additional tests 

        Checks:
        Multicollinearity -- Tests for multicollinearity of the design matrix. Inherited from regression base class
        Autocorrelation -- Tests for autocorrelation of the residuals. Inherited from regression base class
        High-Dimensionality -- Checks if the data is high-dimensional (p>>n). Inherited from regression base class
        Singular -- Checks if data matrix is singular. Inherited from regression base class
        Linearity -- Tests if linearity assumption of OLS appears justified
        Gaussian Residuals -- Tests if the residuals appear to come from a Gaussian distribution
        """
        rc.REG.check_model(self)
        self.linCheck = c.linCheck(self.independentVar, self.dependentVar)
        self.linCheck.check()
        self.normCheck = c.normCheck(self.residuals)
        self.normCheck.check()
        self.homoskeCheck = c.homoskeCheck(self.residuals, self.independentVar)
        self.homoskeCheck.check()

    def plot_results(self):
        """Creates the base regression plots as well as a qq-Plot

        Plots:
        Residual Plots -- plot of residuals vs fitted values. Inherited from regression base class
        QQ-Plot -- plot of the residual empirical quantiles against those of a normal distribution 
        """
        rc.REG.plot_results(self)
        viz.plot_qq(self.residuals).plot()
