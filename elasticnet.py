from sklearn import linear_model
import statsmodels.api as sm

class ELASTICNET:
    #Implements elastic-net regression, with assumption checks

    def __init__(self, indepVar, depVar,alpha=1, l1_ratio=.5):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, copy_X=False)

    def fit_model(self):
        self.fitted_model = self.regObj.fit(self.independentVar, self.dependentVar)
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def elasticnet_CV(self):
        #Chooses regularization parameter using cross-validation
        self.regObj = linear_model.ElasticNetCV()
        self.fit_model()
        self.regObj = linear_model.ElasticNet(alpha=self.regObj.alpha_, l1_ratio=self.regObj.l1_ratio_, copy_X=False)

    def checks(self):
        #Variety of checks for elastic net fit
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
