from sklearn import linear_model
import statsmodels.api as sm
import checks as c
import boxcox as bc
import lasso as lasso
import elasticnet as elasticnet
import ridge as ridge
import numpy as np
import visualizations as viz
import regClass as rc

class OLS(rc.REG):
    """Object which performs ordinary least squares regression, checks assumptions, and makes plots

    Methods:
    __init__ -- inherits from regression base class
    fit_model -- inherits from regression base class
    check_model
    print_results -- inherits from regression base class
    plot_results
    model_actions

    Instance Variables:
    self.regObj -- primary regression object, from scikit library
    self.residuals -- vector, 1xnObservations, containing residuals of fit
    """

    def check_model(self):
        """Checks assumptions of OLS regression. Inherits 4 from regression base class and adds 3 additional tests 

        Checks:
        Multicollinearity -- Tests for multicollinearity of the design matrix. Inherited from regression base class
        Autocorrelation -- Tests for autocorrelation of the residuals. Inherited from regression base class
        High-Dimensionality -- Checks if the data is high-dimensional (p>>n). Inherited from regression base class
        Singular -- Checks if data matrix is singular. Inherited from regression base class
        Linearity -- Tests if linearity assumption of OLS appears justified
        Gaussian Residuals -- Tests if the residuals appear to come from a Gaussian distribution
        """
        rc.REG.check_model(self)
        self.linCheck = c.linCheck(self.independentVar, self.dependentVar)
        self.linCheck.check()
        self.normCheck = c.normCheck(self.residuals)
        self.normCheck.check()
        self.homoskeCheck = c.homoskeCheck(self.residuals, self.independentVar)
        self.homoskeCheck.check()

    def plot_results(self):
        """Creates the base regression plots as well as a qq-Plot

        Plots:
        Residual Plots -- plot of residuals vs fitted values. Inherited from regression base class
        QQ-Plot -- plot of the residual empirical quantiles against those of a normal distribution 
        """
        rc.REG.plot_results(self)
        viz.plot_qq(self.residuals).plot()

    def model_actions(self):
        self.acAction()
        self.mcAction()
        self.linAction()
        self.highdimAction()
        self.singAction()
        self.homoskeAction()

    def mcAction(self):
        if self.mcCheck.conNum > 20:
            print "Multicollinearity is a problem. Condition number of design matrix is  " , self.mcCheck.conNum
            if self.sparse == True:
                print "The underlying model is also sparse. Fitting elastic-net regression."
                return elasticnet.ELASTICNET(self.independentVar, self.dependentVar, alpha=.5, l1_ratio=.5)
                #Possibly PCA regression?
            else:
                print "The underlying model isn't sparse. Fitting ridge regression. "
                return ridge.RIDGE(self.independentVar, self.dependentVar, alpha=1)
        elif self.highdimCheck == True:
            print "Multicollinearity is not an issue, but the data is high dimensional. Fitting lasso and orthogonal matching pursuit regression. "
            return lasso.LASSO(self.independentVar, self.dependentVar)
            #self.newObj2 = omp.OMP(self.independenVar, self.dependentVar)
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
            print ("Singular data matrix. Inspect data and remove linearly dependent samples.")

    def homoskeAction(self):
        if self.homoskeCheck.bptest[1] < .05:
            print "Evidence of heteroskedasticity. Use only robust standard errors."
        else:
            print "Heteroskedasticity does not appear to be a problem."
