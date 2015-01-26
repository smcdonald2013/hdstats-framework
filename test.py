import ols
import lasso
import numpy as np
#import statsmodels.stats.outliers_influence as oi
import boxcox as bc
import statsmodels.api as sm
import dataAnalysis
import hdstats-framework

dep = np.array([0, 1, 4, 9, 16, 25])
indep = np.array([[0, 0], [1, 1], [2, 1], [3, 5], [4, 5], [5, 7]])

#Practice data
#nsample = 100
#x = np.linspace(0, 10, 100)
#X = np.column_stack((x, x**2))
#beta = np.array([1, .1, 10])
#e = np.random.normal(size=nsample)
#X = sm.add_constant(X, prepend=False)
#y = np.dot(X, beta) + e

nobs = 100
X = np.random.random((nobs, 3))
X[:,1] = X[:,2]
beta = [1, .5, .5]
e = np.random.random(nobs)
y = np.dot(X, beta) + e

dep = y
indep = X

data = np.concatenate((y,X), axis=1)

dataset = Dataset(data=data)

dataAnalysis.main(dataset)



print dep
print indep
#print oi.variance_inflation_factor(dep,0)
#print oi.variance_inflation_factor(dep,1)

trans = np.empty([indep.shape[0],indep.shape[1]])
for i in range(indep.shape[1]):
    linData = bc.LINTRANS(indep[:,i], dep)
    linData.linearize()
    trans[:,i] = linData.opTrans
    print linData.xlam

x = ols.OLS(indep, dep)

x.fit_model()

print x.fitted_model

print x.regObj.coef_

print x.residuals

x.checks()

#print x.mcCheckOutput

#print x.linCheck.hc

x.actions()

#print x.acCheck

y = lasso.LASSO(indep,dep)

y.fit_model()

print y.fitted_model

print y.regObj.coef_

y.checks()
