from sklearn import linear_model
import statsmodels.api as sm

class ELASTICNET:
    #Implements ridge regression, with assumption checks

    def __init__(self, indepVar, depVar,alpha, l1_ratio):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, copy_X=False)

    def fit_model(self):
        self.fitted_model = self.regObj.fit(self.dependentVar, self.independentVar)
        self.residuals = self.independentVar - self.regObj.decision_function(self.dependentVar)

    def elasticnet_CV(self):
        self.regObj = linear_model.ElasticNetCV()
        self.fit_model()
        self.regObj = linear_model.Lasso(alpha=self.regObj.alpha_, l1_ratio=self.regObj.l1_ratio_, copy_X=False)

    def vif(self):
        #This doesn't currently work, it appears vif may not yet be implemented in stats models
        self.vif = []
        for i in range(len(self.dependentVar)):
            self.vif.append(sm.stats.outliers_influence.variance_inflation.factor(self.dependentVar,i)) 

    def checks(self):
        self.acCheck = sm.stats.stattools.durbin_watson(self.residuals)
        self.normCheck = sm.stats.diagnostic.normal_ad(self.residuals)
        #self.mcCheck = self.vif()
