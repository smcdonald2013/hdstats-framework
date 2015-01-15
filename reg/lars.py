"""@package LARS
The elastic net module, which performs a box-cox transformation."""
from sklearn import linear_model
import visualizations as viz
import checks as c
import regClass as rc

class LARS(rc.REG):
    """Object which performs LARS, checks assumptions, and makes plots."""

    def __init__(self, indepVar, depVar):
        """LARS constructor

        @param indepVar Array of independent variables
        @param depVar vector of dependent varible
        """
        rc.REG.__init__(self, indepVar, depVar)
        ## LARS Object (from Scikit-learn)
        self.regObj = linear_model.Lars()

    def CV_model(self):
        """Perform cross-validation to select the correct alpha parameters for LARS"""
        self.regObj = linear_model.LarsCV()

    def plot_results(self):
        """Create the base regression plots as well as a qq-plot"""
        rc.REG.plot_results(self)
        viz.plot_qq(self.residuals).plot()
