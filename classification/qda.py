from sklearn.qda import QDA as skQDA
import checks as c
import numpy as np
import visualizations as viz
import classifierBase as cb

class QDA(cb.CLASSIFIER):
    """Classifier using quadratic discriminant analysis, with associated checks and plots."""

    def __init__(self, data, classes, classNames=False):
        """Uses base initilization, but with QDA object.
        
        @param data Data Array
        @param classes Vector of class labels
        @param classNames Vector of class names
        """
        cb.CLASSIFIER.__init__(self, data, classes, classNames)
        ## QDA Class Object (from Scikit-learn)
        self.classObj = skQDA()

    def check_model(self):
        """Inherits multicollinearity check from base, and adds multivariate normal check. """
        cb.CLASSIFIER.check_model(self)
        ## Multivariate normal check class object
        self.mvnCheck = c.mvnCheck(self.data)
        self.mvnCheck.check()

    def print_results(self):
        """Overwrites base classifier because QDA does not have coefficient estimates."""
        print('\n Class Means')
        print(self.classObj.means_)

    def plot_results(self):
        """Overwrites base plots, because of odd QDA implementation in scikit-learn."""
        viz.plot_clusters(self.data[:,0],self.data[:,1],self.classes).plot()
