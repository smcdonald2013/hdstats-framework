import ols

indep = [0, 1, 2]
dep = [[0, 0], [1, 1], [2, 2]]

x = ols.OLS(indep, dep)

x.fit_model()

print x.fitted_model
