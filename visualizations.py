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

class plot_comps:
    #2-D plot of one component vs another
    def __init__(self,comp1, comp2, compNums, classes=False, classNames=False):
        self.comp1 = comp1
        self.comp2 = comp2
        self.compNums = compNums
        self.classes = classes
        self.classNames = classNames

    def plot(self):
        if self.classes==False:
            plt.scatter(comp1, comp2)
            plt.xlabel('Component ', compNums[0])
            plt.ylabel('Component ', compNums[1])
            plt.show()
        else:
            for i, j, class_name in zip('rgb', [1,2,3], self.classNames):
                plt.scatter(self.comp1[self.classes == j], self.comp2[self.classes == j], c=i, label=class_name)
            plt.legend()
            plt.show()
