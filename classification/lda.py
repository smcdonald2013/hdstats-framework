from sklearn.lda import LDA as skLDA
import checks as c
import logistic
import visualizations as viz
import classifierBase as cb

class LDA(cb.CLASSIFIER):
    """Classifier using linear discriminant analysis, with associated checks and plots. """

    def check_model(self):
        """Inherits multicollinearity check from base, and adds multivariate normal check"""
        cb.CLASSIFIER.check_model(self)
        ## Multivariate normal check object
        self.mvnCheck = c.mvnCheck(self.data)
        self.mvnCheck.check()
