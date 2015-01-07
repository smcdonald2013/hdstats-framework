import matplotlib.pyplot as plt
import statsmodels.api as sm

class plot_residuals:
    #Plots the residuals

    def __init__(self, residuals, fittedValues):
        self.residuals = residuals
        self.fittedValues = fittedValues

    def plot(self):
        plt.scatter(self.fittedValues, self.residuals, color='black')
        plt.xticks(())
        plt.yticks(())
        plt.show()

class plot_regPath:
    #Plots the regularization path (coefficients as a function of the regularization parameter. Typically used for ridge/lasso 
    def __init__(self, alphas, coefs):
        self.alphas = alphas
        self.coefs = coefs

    def plot(self):
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
    #Plots the vector given against the quantiles of a theoretical normal distribution. Normally, this should receive the residuals of a regregression as input. 
    def __init__(self, data):
        self.data = data

    def plot(self):
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
 
