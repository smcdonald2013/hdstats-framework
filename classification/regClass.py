from sklearn import linear_model
import visualizations as viz
import checks as c

class REG:
    #Base class for regression analysis

    def __init__(self, indepVar, depVar, sparse=False):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.LinearRegression() ###This should always be overridden 
        self.sparse = sparse

    def fit_model(self):
        self.fitted_model = self.regObj.fit(self.independentVar, self.dependentVar)
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def check_model(self):
        self.acCheck = c.acCheck(self.residuals)
        self.acCheck.check()
        self.mcCheck = c.mcCheck(self.independentVar, self.dependentVar, self.residuals)
        self.mcCheck.check()
        self.singCheck = c.singCheck(self.independentVar)
        self.singCheck.check()
        self.highdimCheck = c.highdimCheck(self.independentVar)
        self.highdimCheck.check()

    def print_results(self):
        print('\n Regression Coefficients')
        print(self.regObj.coef_)
        print('\n R-Squared')
        print(self.regObj.score(self.independentVar, self.dependentVar))

    def plot_results(self):
        viz.plot_residuals(self.residuals,self.regObj.predict(self.independentVar)).plot()
