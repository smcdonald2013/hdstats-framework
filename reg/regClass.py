"""@package docstring
The base regression module, from which all regression is inherited."""
from sklearn import linear_model
import visualizations as viz
import checks as c

class REG:
    """Base class for regression analysis.

    In general, children of this class will override or extend most, but not all of the methods. Some methods will be taken straight from the base implementation if there are no analysis-specific implementations.
    """

    def __init__(self, indepVar, depVar, sparse=False):
        """Base constructor, should always be overwritten or extend for non-OLS, because it creates an OLS object.
        @param indepVar Array of independent variables
        @param depVar Vector of dependent variable"""
        ## Vector of dependent variable
        self.dependentVar = depVar
        ## Array of independent variables
        self.independentVar = indepVar
        ## Regression object, by default OLS, should be overwritten
        self.regObj = linear_model.LinearRegression()
        ##Boolean declaring if model is presumed sparse
        self.sparse = sparse

    def fit_model(self):
        """Fits the model, and finds the residuals."""
        ## Fitted model object
        self.fitted_model = self.regObj.fit(self.independentVar, self.dependentVar)
        ## Residuals for the model
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def check_model(self):
        """Default model checks"""
        ## Autocorrelation Check Object
        self.acCheck = c.acCheck(self.residuals)
        self.acCheck.check()
        ## Multicollinearity Check Object
        self.mcCheck = c.mcCheck(self.independentVar, self.dependentVar, self.residuals)
        self.mcCheck.check()
        ## Singularity Check Object
        self.singCheck = c.singCheck(self.independentVar)
        self.singCheck.check()
        ## High-Dimensionality Check Object
        self.highdimCheck = c.highdimCheck(self.independentVar)
        self.highdimCheck.check()

    def print_results(self):
        """Prints the estimated coefficients and r-squared."""
        print('\n Regression Coefficients')
        print(self.regObj.coef_)
        print('\n R-Squared')
        print(self.regObj.score(self.independentVar, self.dependentVar))

    def plot_results(self):
        """The default plot is the residuals vs fitted values."""
        viz.plot_residuals(self.residuals,self.regObj.predict(self.independentVar)).plot()
