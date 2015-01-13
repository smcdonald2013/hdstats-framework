from sklearn import linear_model
import checks as c
import visualizations as viz
import classifierBase as cb

class LOGISTIC(cb.CLASSIFIER):
    """Classifier using logistic regression, with associated checks and plots.

    Methods:
        __init__ -- Inherited from base classification
        fit_model -- Inherited from base classification
        check_model -- Inherited from base classification
        print_results -- Overwrites base classification
        plot_results -- Inherited from base classification
    """

    def __init__(self, data, classes, penalty='l2',dual=False):
        """Uses base constructor, but sets additional variables for penalty term and dual form."""
        cb.CLASSIFIER.__init__(self, data, classes)
        self.penalty = penalty
        self.dual = dual
        self.classObj = linear_model.LogisticRegression(penalty=self.penalty, dual=self.dual)

    def print_results(self):
        """Completely overrides base classifier print class, as class means is not really applicable to logistic regression. """
        print('\n Logistic Coefficients')
        print(self.classObj.coef_)
