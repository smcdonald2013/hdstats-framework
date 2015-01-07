from sklearn import linear_model
import statsmodels.api as sm

class OMP:
    #Implements orthogonal matching pursuit regression, with assumption checks

    def __init__(self, indepVar, depVar):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.OrthogonalMatchingPursuit()

    def fit_model(self):
        self.fitted_model = self.regObj.fit(X=self.independentVar, y=self.dependentVar)
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def checks(self):
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
