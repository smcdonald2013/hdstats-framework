from sklearn import linear_model
import checks as c
import visualizations as viz
import regClass as rc

class OLS(rc.REG):
    """Object which performs ordinary least squares regression, checks assumptions, and makes plots."""

    def check_model(self):
        """Checks assumptions of OLS regression. Inherits 4 from regression base class and adds 3 additional tests."""
        rc.REG.check_model(self)
        #Linearity Check object
        self.linCheck = c.linCheck(self.independentVar, self.dependentVar)
        self.linCheck.check()
        #Normality Check object
        self.normCheck = c.normCheck(self.residuals)
        self.normCheck.check()
        #Homoskedasticity Check object
        self.homoskeCheck = c.homoskeCheck(self.residuals, self.independentVar)
        self.homoskeCheck.check()

    def plot_results(self):
        """Creates the base regression plots as well as a qq-Plot."""
        rc.REG.plot_results(self)
        viz.plot_qq(self.residuals).plot()
