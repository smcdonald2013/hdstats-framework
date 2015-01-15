"""@package docstring
GLSAR module, for models with autocorrelated error terms."""
import regClass as rc 
import statsmodels as sm

class GLSAR(rc.REG):
    """Performs generalized least squares autocorrelated residuals regression, as implemented by statsmodels."""
    
    def __init__(self, indepVar, depVar, coeff=1):
        """Uses the baseclass initialization, with additional coefficient parameter used for the GLSAR model.

        @param indepVar Array of independent variables
        @param depVar Vector of the endogenous variable
        @param coeff Coefficient used in GLSAR (see statsmodels documentation)
        """
        rc.REG.__init__(self, indepVar, depVar)
        ## Coefficient for model fit (see statsmodels documentation)
        self.coeff = coeff
        ## GLSAR object, from statsmodels
        self.regObj = sm.regression.linear_model.GLSAR(depVar, indepVar, rho=coeff)

    def fit_model(self):
        """Fits the model, no residuals obtained.

        Statsmodels GLSAR function does not appear to have way to obtain the residuals, so that is not implemented, limiting the functionality of later methods"""
        ##Fitted GLSAR object, from statsmodels
        self.fitted_model = self.regObj.fit()

    def check_model(self):
        """Informs the user that without residuals, checks cannot be performed."""
        print('\n Statsmodels does not provide a way to obtain the residuals from the GLSAR fit, so post-fitting checks cannot be performed.')

    def print_results(self):
        """Print the results uses a slightly different format than is implemented in base class, due to statsmodels."""
        print('\n Regression Coefficients')
        print(self.fitted_model.params)

    def plot_results(self):
        """Informs the values that standard plots cannot be created"""
        print('\n Due to statsmodels limitations, standard analysis plots cannot be created for GLSAR.')
