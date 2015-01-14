import sys, unittest
import numpy as np
from reg import ols, ridge, lasso
from classification import lda, qda, logistic
from sklearn import datasets

class RegressionTest(unittest.TestCase):
    """Tests the regression coefficients."""

    def setUp(self):
        """Establish the test"""
        self.dep = np.array([0, 1, 4, 9, 16, 25])
        self.indep = np.array([[0, 0], [1, 1], [2, 1], [3, 5], [4, 5], [5, 7]])

    def tearDown(self):
        """Cleanup"""
        del self.dep
        del self.indep

    def testOLS(self):
        """Test OLS coefficients, correct answers obtained from R."""
        olsObj = ols.OLS(self.indep, self.dep)
        olsObj.fit_model()
        correctAns = np.array([3.68, .91])
        npTest = np.testing.assert_array_almost_equal(olsObj.fitted_model.coef_, correctAns, decimal=2)
        self.assertEqual(npTest, None)

    def testRidge(self):
        """Test Ridge coefficients."""
        ridgeObj = ridge.RIDGE(self.indep, self.dep)
        ridgeObj.fit_model()
        correctAns = np.array([2.62, 1.53])
        npTest = np.testing.assert_array_almost_equal(ridgeObj.fitted_model.coef_, correctAns, decimal=2)
        self.assertEqual(npTest, None)

    def testLasso(self):
        """Test Lasso coefficients."""
        lassoObj = lasso.LASSO(self.indep, self.dep)
        lassoObj.fit_model()
        correctAns = np.array([1.30, 2.02])
        npTest = np.testing.assert_array_almost_equal(lassoObj.fitted_model.coef_, correctAns, decimal=2)
        self.assertEqual(npTest, None)


class ClassificationTest(unittest.TestCase):
    """Tests the classification algorithms using the iris dataset."""

    def setUp(self):
        """Establish the test"""
        irisData = datasets.load_iris()
        self.classes = irisData.target
        self.data = irisData.data

    def tearDown(self):
        """Cleanup"""
        del self.classes
        del self.data

    def testLDA(self):
        """Test LDA , correct answers obtained from R."""
        ldaObj = lda.LDA(self.data, self.classes)
        ldaObj.fit_model()
        correctAns = np.array([5.01, 3.42, 1.46, .24])
        npTest = np.testing.assert_array_almost_equal(ldaObj.fitted_model.means_[0], correctAns, decimal=2)
        self.assertEqual(npTest, None)

    def testQDA(self):
        """Test QDA , correct answers obtained from R."""
        qdaObj = qda.QDA(self.data, self.classes)
        qdaObj.fit_model()
        correctAns = np.array([5.01, 3.42, 1.46, .24])
        npTest = np.testing.assert_array_almost_equal(qdaObj.fitted_model.means_[0], correctAns, decimal=2)
        self.assertEqual(npTest, None)

if __name__ == "__main__":
    unittest.main()
