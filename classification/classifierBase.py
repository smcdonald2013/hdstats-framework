from sklearn.lda import LDA as skLDA
import visualizations as viz
import checks as c

class CLASSIFIER:
    """This is the base class for classification algorithms."""

    def __init__(self, data, classes, classNames=False):
        """Initializes the classifier. Accepts class names as optional argument.
        
        @param data Data array
        @param classes Vector of class labels
        @param classNames Vector of class names"""
        ## Data array
        self.data = data
        ## Vector of classes
        self.classes = classes
        ## Class object, LDA by default
        self.classObj = skLDA()
        ## Vector of class names
        self.classNames = classNames

    def fit_model(self):
        """Fits the classifier using scikit-learn method."""
        ## Scikit learn fitted model object
        self.fitted_model = self.classObj.fit(self.data, self.classes)

    def check_model(self):
        """Base model check is just for mutlicollinearity."""
        ## Multicollinearity check object
        self.mcCheck = c.mcCheck(self.data)
        self.mcCheck.check()

    def print_results(self):
        """Prints the fitted coefficients and the class means."""
        print('\n Classifier Coefficients')
        print(self.classObj.coef_)
        print('\n Class Mean')
        print(self.classObj.means_)

    def plot_results(self):
        """By default, we plot the first two variables colored by class."""
        ## Array of data transformed by analysis method
        self.transData = self.fitted_model.transform(self.data)
        viz.plot_clusters(self.transData[:,0],self.transData[:,1], self.classes).plot()
