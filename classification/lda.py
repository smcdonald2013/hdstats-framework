from sklearn.lda import LDA as skLDA
import checks as c
import logistic
import visualizations as viz
import classifierBase as cb

class LDA(cb.CLASSIFIER):
    """Classifier using linear discriminant analysis, with associated checks and plots.

    Methods:
        __init__ -- Inherited from base classification
        fit_model -- Inherited from base classification
        check_model -- Adds multivariate normality check
        print_results -- Inherited from base classification
        plot_results -- Inherited from base classification
    """

    def check_model(self):
        """Inherits multicollinearity check from base, and adds multivariate normal check"""
        cb.CLASSIFIER.check_model(self)
        self.mvnCheck = c.mvnCheck(self.data)
        self.mvnCheck.check()
        #self.eqCov = c.eqCovCheck(self.data, self.classes)
        #self.eqCov.check()

    def action_model(self):
        self.mvnAction()
        #self.eqCovAction()

    def mvnAction(self):
        if self.mvnCheck.skewp < .05:
            print "Data does not appear to have come from a multivariate normal distribution. Logistic Regression may be more appropriate. Running now:."
            return logistic.LOGISTIC(self.data, self.classes)
        else:
            print "Data appear to be multivariate-normal."

    def eqCovAction(self):
        if self.eqCov.boxMp < .05:
            print "Classes do not appear to have similar covariance matrices. QDA would be more appropriate, performing now."
            return qda.QDA(self.data, self.classes)
        else:
            print "Classes appear to have similar covariance matrices, LDA may be justified."


