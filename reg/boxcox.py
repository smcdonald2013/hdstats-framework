import scipy as sc
import numpy as np
import statsmodels as sm

class LINTRANS:

    def __init__(self, indepVar, depVar):
        self.dependentVar = depVar
        self.independentVar = indepVar
        
    def linearize(self):
        self.lam = np.arange(0,2,.1)
        self.cor = np.empty(self.lam.shape[0])
        self.shift = min(self.independentVar)
        if self.shift <= 0:
            self.independentVar = self.independentVar - self.shift + .0001
        for i in range(self.lam.shape[0]):
            self.transVar = sc.stats.boxcox(x=self.independentVar, lmbda=self.lam[i])
            self.cor[i] = np.corrcoef(self.transVar, self.dependentVar)[0,1]
        self.max = max(self.cor)
        self.maxIndex = np.where(self.cor==self.max)
        print self.lam[self.maxIndex]
        self.opTrans = sc.stats.boxcox(x=self.independentVar, lmbda=self.lam[self.maxIndex])
        self.xlam = self.lam[self.maxIndex]

