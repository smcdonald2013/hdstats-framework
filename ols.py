from sklearn import linear_model
import statsmodels.api as sm
import checks as c

class OLS:
    #Implements ordinary least squares regression, with assumption checks

    def __init__(self, indepVar, depVar):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.LinearRegression()

    def fit_model(self):
        self.fitted_model = self.regObj.fit(self.dependentVar, self.independentVar)
        self.residuals = self.independentVar - self.regObj.decision_function(self.dependentVar)

    def checks(self):
        self.acCheck = c.acCheck(self.residuals)
        self.linCheck = c.linCheck(self.independentVar, self.dependentVar, self.residuals)
        self.normCheck = sm.stats.diagnostic.normal_ad(self.residuals)
        self.mcCheck = c.mcCheck(self.independentVar, self.dependentVar, self.residuals)
        self.mcCheck.check()
        self.linCheck.check()

    def actions(self):
        #self.acAction(acCheck)
        self.mcAction()

    def mcAction(self):
        for i in range(self.dependentVar.shape[0]):
            if self.mcCheck.vif[i] > 4:
                print "Multicollinearity at ", i
