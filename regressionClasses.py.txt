class OLS
        ###This initializes any necessary variables, and creates the regression object
	def_init_(self,depVar, indepVar):
		self_dependentVars = depVar
                        self_independentVars = indepVar
		self_regObj = linear_model.LinearRegression()
	###implementation fits the scikit model
	def implementation():
		fittedObject = regObj.fit(depVar,indepVar)

	###performs checks. Checks are defined elsewhere (they are subclasses of generic 'check class')
	def checks()
		acCheckOuput = acCheck.test(fittedObject.resid)
		normCheckOutput = normCheck.test(fittedObject.resid)
		mcCheckOutput = mcCheck.test(fittedObject.resid)
		hdCheckOutput = hdCheck.test(fittedObject.resid)
		###Other checks to add- linearity, model stability, others

	def actions(params)
		acAction(params)
		normAction(params)
		mcAction(params)
		hdActionparams)

class Ridge
        ###This initializes any necessary variables, and creates the regression object
	def_init_(self,depVar, indepVar):
		self_dependentVars = depVar
                             self_independentVars = indepVar
		self_regObj = linear_model.RidgeRegression()
	###implementation fits the scikit model
	def implementation():
		fittedObject = regObj.fit(depVar,indepVar)

	###performs checks. Checks are defined elsewhere (they are subclasses of generic 'check class')
	def checks()
		acCheckOuput = acCheck.test(fittedObject.resid)
		normCheckOutput = normCheck.test(fittedObject.resid)
		mcCheckOutput = mcCheck.test(fittedObject.resid)
		hdCheckOutput = hdCheck.test(fittedObject.resid)
		###Other checks to add- linearity, model stability, others

	def actions(params)
		acAction(params)
		normAction(params)
		mcAction(params)
		hdActionparams)

Class acCheck
	def_init(self, data)
		self_resid = data

	def test()
		durbin_watson(resid)

Class normCheck
	def_init(self, data)
		self_resid = data

	def test()
		kstest_normal(resid)

Class hdCheck
	def_init(self, data)
		self_resid = data

	def test()
		numVar/numSamp
