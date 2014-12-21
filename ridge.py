from sklearn import linear_model
import statsmodels.api as sm

class RIDGE:
    #Implements ridge regression, with assumption checks

    def __init__(self, indepVar, depVar,alpha=1):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.Ridge(alpha=alpha, copy_X=False)

    def fit_model(self):
        self.fitted_model = self.regObj.fit(self.independentVar, self.dependentVar)
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def ridge_CV(self):
        #Fits ridge using CV
        self.regObj = linear_model.RidgeCV()
        self.fit_model()
        self.regObj = linear_model.Ridge(alpha=self.regObj.alpha_, copy_X=False)

def checks(self):
        #Variety of checks for ridge fit
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
