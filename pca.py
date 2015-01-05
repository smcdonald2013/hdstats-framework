from sklearn import decomposition

class PCA:
    # Implements Principal Component Analysis


    def __init__(self, data):
        self.data = data
        self.dataTransformed = None
        self.decObj = decompostion.RandomizedPCA()

    def fit_model(self):
        self.dataTransformed = self.decObj.fit_transform(self.data)
