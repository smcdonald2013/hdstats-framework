from sklearn import linear_model
import checks as c
import visualizations as viz
import classifierBase as cb

class LOGISTIC(cb.CLASSIFIER):
    """Classifier using logistic regression, with associated checks and plots."""

    def __init__(self, data, classes, penalty='l2',dual=False):
        """Uses base constructor, but sets additional variables for penalty term and dual form.

        @param data Data array
        @param classes Vector of class labels
        @param penalty Regularization penalty applied
        @param dual Boolean for dual or primal formulation"""
        cb.CLASSIFIER.__init__(self, data, classes)
        self.classObj = linear_model.LogisticRegression(penalty=penalty, dual=dual)

    def print_results(self):
        """Completely overrides base classifier print class, as class means is not really applicable to logistic regression. """
        print('\n Logistic Coefficients')
        print(self.classObj.coef_)
