import ols
import lasso

indep = [0, 1, 2]
dep = [[0, 0], [1, 1], [2, 2]]

x = ols.OLS(indep, dep)

x.fit_model()

print x.fitted_model

print x.regObj.coef_

x.checks()

#print x.acCheck

y = lasso.LASSO(indep,dep)

y.fit_model()

print y.fitted_model

print y.regObj.coef_

y.checks()
