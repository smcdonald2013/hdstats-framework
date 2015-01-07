from sklearn import linear_model
import statsmodels.api as sm

class LARS:
    #Implements LARS regression, with assumption checks

    def __init__(self, indepVar, depVar,alpha, l1_ratio):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.Lars()

    def fit_model(self):
        self.fitted_model = self.regObj.fit(X=self.independentVar, y=self.dependentVar)
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def CV_model(self):
        #Chooses regularization parameter using cross-validation
        self.regObj = linear_model.LarsCV()

    def check_model(self):
        #Variety of checks for least angle regression fit
        self.acCheck = c.acCheck(self.residuals)
        self.acCheck.check()
        self.linCheck = c.linCheck(self.independentVar, self.dependentVar)
        self.linCheck.check()
        self.normCheck = c.normCheck(self.residuals)
        self.normCheck.check()

    def print_results(self):
        #
        print('\n Lars Coefficients')
        print(self.regObj.coef_)
        print('\n R-Squared')
        print(self.regObj.score(self.independentVar, self.dependentVar))
