from sklearn import linear_model
import statsmodels.api as sm
import checks as c

class TIMESERIES:
    #Implements ordinary least squares regression, with assumption checks

    def __init__(self, indepVar, depVar,order=np.array([1,1]):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.order = order
        self.regObj = sm.statsmodels.tsa.arima_model.ARMA(endog=self.dependentVar, order=self.order, exog=self.independentVar)

    def fit_model(self):
        self.fitted_model = self.regObj.fit()
        self.residuals = self.regObj.geterrors()

    def timeseries_order(self):
        self.orderObj = sm.tsa.arma_order_select_ic(self.residuals)
        self.min_order = self.orderObj.bic_min_order

    def checks(self):
        self.acCheck = c.acCheck(self.residuals)
        self.normCheck = sm.stats.diagnostic.normal_ad(self.residuals)
        self.mcCheck = c.mcCheck(self.independentVar, self.dependentVar, self.residuals)
        self.mcCheck.check()

    def actions(self):
        #self.acAction(acCheck)
        self.mcAction()

    def mcAction(self):
        for i in range(self.dependentVar.shape[0]):
            if self.mcCheck.vif[i] > 4:
                print "Multicollinearity at ", i
