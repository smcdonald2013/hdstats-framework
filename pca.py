from sklearn import decomposition

class RPCA:
    # Implements Principal Component Analysis

    def __init__(self, data):
        self.data = data
        self.dataTransformed = None
        self.obj = decomposition.RandomizedPCA()

    def fit_model(self):
        self.dataTransformed = self.obj.fit_transform(self.data)
