from sklearn import linear_model

class OLS:
    #Implements ordinary least squares regression, with assumption checks

    def __init__(self, indepVar, depVar):
        self.dependentVar = depVar
        self.independentVar = indepVar
        self.regObj = linear_model.LinearRegression()

    def fit_model(self):
        self.fitted_model = self.regObj.fit(self.dependentVar, self.independentVar)
