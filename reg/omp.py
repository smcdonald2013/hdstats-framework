"""@package docstring
The elastic net module, which combines l1 and l2 penalties."""
from sklearn import linear_model
import checks as c
import visualizations as viz
import regClass as rc

class OMP(rc.REG):
    """Object which performs orthogonal matching pursuit regression, checks assumptions, and makes plots."""

    def __init__(self, indepVar, depVar):
        """OMP constructor

        @param indepVar Array of independent variables
        @param depVar Vector of dependent variable
        """
        rc.REG.__init__(self, indepVar, depVar)
        ## OMP Object (from Scikit-learn)
        self.regObj = linear_model.OrthogonalMatchingPursuit()

    def plot_results(self):
        """Create the base regression plots as well as a qq-plot"""
        rc.REG.plot_results(self)
        viz.plot_qq(self.residuals).plot()
