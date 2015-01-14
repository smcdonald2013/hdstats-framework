import regClass as rc 
import statsmodels as sm

class GLSAR(rc.REG):
    """Object which performs generalized least squares regression. Intended to be used only after another regression has been run, which found autocorrelation in the residuals.

    Methods:
    __init__ -- extends regression base class
    fit_model -- overwrites regression base class
    check_model -- inherits from regression base class
    print_results -- Overwrites regression base class
    plot_results -- inherits from regression base class

    Instance Variables:
    self.regObj -- primary regression object, from scikit library
    self.residuals -- vector, 1xnObservations, containing residuals of fit
    """
    def __init__(self, indepVar, depVar, coeff=1):
        """Uses the baseclass initialization, with additional coefficient parameter used for the GLSAR model.
        """
        rc.REG.__init__(self, indepVar, depVar)
        self.coeff = coeff
        self.regObj = sm.regression.linear_model.GLSAR(depVar, indepVar, rho=coeff)

    def fit_model(self):
        """Since GLSAR is a statsmodels function, retrieving the residuals is slightly different than in scikit-learn regressions.."""
        self.fitted_model = self.regObj.fit()
        self.residuals = self.dependentVar - self.fitted_model.fittedvalues
        print(self.residuals)

    def print_results(self):
        """Print the results uses a slightly different format than is implemented in base class, due to statsmodels."""
        print('\n Regression Coefficients')
        print(self.fitted_model.params)
