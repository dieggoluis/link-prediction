from sklearn.base import BaseEstimator
import xgboost as xgb

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = xgb.XGBClassifier()

    def fit(self, X, y):
        print 'fit classifier...'
        self.clf.fit(X, y)

    def predict(self, X):
        print 'predict classifier...'
        return self.clf.predict(X)
