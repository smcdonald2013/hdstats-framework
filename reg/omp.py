from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import visualizations as viz

class OMP:
    #Implements orthogonal matching pursuit regression, with assumption checks

    def __init__(self, indepVar, depVar):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.OrthogonalMatchingPursuit()

    def fit_model(self):
        self.fitted_model = self.regObj.fit(X=self.independentVar, y=self.dependentVar)
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def check_model(self):
        #Variety of checks for omp fit
        self.acCheck = c.acCheck(self.residuals)
        self.acCheck.check()
        self.linCheck = c.linCheck(self.independentVar, self.dependentVar)
        self.linCheck.check()
        self.normCheck = c.normCheck(self.residuals)
        self.normCheck.check()
        self.mcCheck = c.mcCheck(self.independentVar, self.dependentVar, self.residuals)
        self.mcCheck.check()
        self.homoskeCheck = c.homoskeCheck(self.residuals, self.independentVar)
        self.homoskeCheck.check()

    def print_results(self):
        print('\n OMP Coefficients')
        print(self.regObj.coef_)
        print('\n R-Squared')
        print(self.regObj.score(self.independentVar, self.dependentVar))

    def plot_results(self):
        viz.plot_residuals(self.residuals,self.regObj.predict(self.independentVar)).plot()
        viz.plot_qq(self.residuals).plot()
