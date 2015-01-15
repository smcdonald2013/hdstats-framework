"""@package docstring
The visualizations module, which contains a variety of visualization classes.

This is the location of all of the visualizations used in the project. 
There was enough variety in the visualizations, that creating a base class
seemed unecessary, however that may be an area for expansion in the future. 
"""

import matplotlib.pyplot as plt
import statsmodels.api as sm

class plot_residuals:
    """Basic plot of residuals vs fitted values."""

    def __init__(self, residuals, fittedValues):
        """Constructor, just sets member variables.

        @param residuals Vector of residuals.
        @param fittedValues Vector of fitted values.
        """
        ##Vector of residuals
        self.residuals = residuals
        ##Vector of fitted values
        self.fittedValues = fittedValues

    def plot(self):
        """Makes the plot using matplotlib functions."""
        plt.scatter(self.fittedValues, self.residuals, color='black')
        plt.xticks(())
        plt.yticks(())
        plt.show()

class plot_regPath:
    """Plots the regularization path (coefficients as a function of the regularization parameter.
    
    Typically used for ridge/lasso. 
    """

    def __init__(self, alphas, coefs):
        """Constructor, sets member variables
        
        @param alphas Vector of regularization parameters
        @param coefs Array of coefficients
        """
        ##Vector of regularization parameters
        self.alphas = alphas
        ###Array of coefficients
        self.coefs = coefs

    def plot(self):
        """Produces the plots."""
        ax = plt.gca()
        ax.set_color_cycle(['b','r','g','c','k','y','m'])

        ax.plot(self.alphas, self.coefs)
        ax.set_xscale('log')
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Coefficients as a function of the regularization paramter')
        plt.axis('tight')
        plt.show()

class plot_qq:
    """Plots the vector given against the quantiles of a theoretical normal distribution.

    Normally used with residuals from a regression.
    """
    def __init__(self, data):
        """Constructor, sets member variables
        
        @param data Vector of input data
        """
        ##Vector of input data
        self.data = data

    def plot(self):
        """Makes the plot."""
        sm.qqplot(self.data, fit=True, line='s')
        plt.show()

class crossplot_components:
    #Cross-plot the two first principal components from PCA or similar 
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def plot(self):
        ax = plt.gca()
        ax.scatter(self.c1, self.c2)

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('Cross-plot of two highest-weighted components')
        plt.axis('tight')
        plt.show()

class plot_clusters:
    #Cross-plot the two first variables, colored by cluster index
    def __init__(self, x, y, cluster_index):
        self.x = x
        self.y = y
        self.cluster_index = cluster_index

    def plot(self):
        ax = plt.gca()
        ax.scatter(self.x, self.y, c=self.cluster_index)

        plt.xlabel('Variable 1')
        plt.ylabel('Variable 2')
        plt.axis('tight')
        plt.show()
