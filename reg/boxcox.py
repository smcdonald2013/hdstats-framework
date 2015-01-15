"""@package docstring
The boxcox module, which performs a box-cox transformation. 

This module exists because the boxcox class in stastmodels is not particularly useful. It's output must be manipulated to actually be used in a pratical setting. This module does that manipulation. 
"""

import scipy as sc
import numpy as np
import statsmodels as sm

class LINTRANS:
    """Class to assist with linear transformation of data."""
    
    def __init__(self, indepVar, depVar):
        """Class constructor.

        @param indepVar vector of independent variables, only one variable should be inputted
        @param depVar vector of dependent variable
        """
        ##Vector of dependent variable
        self.dependentVar = depVar
        ##Vector of independent variable
        self.independentVar = indepVar
        
    def linearize(self):
        """This function performs a boxcox transformation on the given data.

        A variety of lambda values are tried, and the one which maximizes the correlation between the transformed data and the dependent variable is selected. 
        """
        lam = np.arange(0,2,.1) #Lambda values to try
        cor = np.empty(lam.shape[0]) #Array to store correlations
        shift = min(self.independentVar) #Independent variable must be postive
        if shift <= 0:
            self.independentVar = self.independentVar - shift + .0001
        for i in range(lam.shape[0]):
            transVar = sc.stats.boxcox(x=self.independentVar, lmbda=lam[i]) #Transformed independent variable
            cor[i] = np.corrcoef(transVar, self.dependentVar)[0,1]
        maxCor = max(cor)
        maxIndex = np.where(cor==maxCor)
        ## The transformed independent variable
        self.opTrans = sc.stats.boxcox(x=self.independentVar, lmbda=lam[maxIndex])
        ## The lambda associated with the optimal transformation
        self.xlam = lam[maxIndex]

