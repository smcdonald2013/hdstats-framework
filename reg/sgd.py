from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import boxcox as bc
import lasso as lasso
import elasticnet as elasticnet
import ridge as ridge
import numpy as np
import visualizations as viz

class SGD:
    #Implements ordinary least squares regression, with assumption checks

    def __init__(self, indepVar, depVar, loss='squared_loss', penalty='l2', alpha = 0, l1_ratio=.5):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.regObj = linear_model.SGDRegressor(loss=self.loss, penalty=self.penalty, alpha=self.alpha, l1_ratio=self.l1_ratio)
        self.sparse = True

    def fit_model(self):
        self.fitted_model = self.regObj.fit(self.independentVar, self.dependentVar)
        self.residuals = self.dependentVar - self.regObj.decision_function(self.independentVar)

    def check_model(self):
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
        self.singCheck = c.singCheck(self.independentVar)
        self.singCheck.check()
        self.highdimCheck = c.highdimCheck(self.independentVar)
        self.highdimCheck.check()

    def print_results(self):
        print('\n OLS Coefficients')
        print(self.regObj.coef_)
        print('\n R-Squared')
        print(self.regObj.score(self.independentVar, self.dependentVar))

    def plot_results(self):
        viz.plot_residuals(self.residuals,self.regObj.predict(self.independentVar)).plot()
        viz.plot_qq(self.residuals).plot()

    def action_model(self):
        self.acAction()
        self.mcAction()
        self.linAction()
        self.highdimAction()
        self.singAction()

    def mcAction(self):
        if self.mcCheck.conNum > 20:
            print "Multicollinearity is a problem. Condition number of design matrix is  " , self.mcCheck.conNum
            if self.sparse == True:
                print "The underlying model is also sparse. Fitting elastic-net regression."
                return SGD(self.independentVar, self.dependentVar, loss="squared_loss", penalty="elasticnet", alpha=.5, l1_ratio=.5)
                #Possibly PCA regression?
            else:
                print "The underlying model isn't sparse. Fitting ridge regression. "
                return SGD(self.independentVar, self.dependentVar, loss="squared_loss", penalty='l2', alpha=1, l1_ratio=0)
        elif self.highdimCheck == True:
            print "Multicollinearity is not an issue, but the data is high dimensional. Fitting lasso regression. There is no SGD implmentation of OMP currently available, so it is not implemented here."
            return SGD(self.independentVar, self.dependentVar, loss="squared_loss", penalty='l1', alpha=1, l1_ratio=1)
            #self.newObj2 = omp.OMP(self.independenVar, self.dependentVar)
            #Also, PCA regression?
        else:
            print "Multicollinearity is not an issue."

    def acAction(self):
        if self.acCheck.ljungbox[1][0] < .05:
            print "Residuals are autocorrelated. Implementing GLSAR."
            #Cochrane-orcutt would be the traditional response. Statsmodels implements GLSAR instead, which appears to be similar. 
            #In the future, adjust this for more than the first lag. 
            return sm.regression.linear_model.GLSAR(self.dependentVar, self.independentVar)
        else:
            print "Residuals appear to be uncorrelated."

    def linAction(self):
        #Transform the variables
        if self.linCheck.hc[1] < .05:
            print "Linear model is incorrect, transforming variables using box-cox transformation"
            self.trans = np.empty([self.independentVar.shape[0],self.independentVar.shape[1]])
            for i in range(self.independentVar.shape[1]):
                self.linData = bc.LINTRANS(self.independentVar[:,i], self.dependentVar)
                #In the future, this should probably be redone using the correlation between the residuals and the independent variables
                self.linData.linearize()
                self.trans[:,i] = self.linData.opTrans
            return  OLS(self.trans, self.dependentVar)
        else:
            print "Linear model appears reasonable."

    def singAction(self):
        if self.singCheck == True:
            print "Singular matrix"
            #Remove linearly dependent rows
