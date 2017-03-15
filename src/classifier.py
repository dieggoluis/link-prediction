from sklearn.base import BaseEstimator
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = xgb.XGBClassifier()
        #self.clf = svm.SVC(kernel='linear', gamma=2)
    def fit(self, X, y):
        print 'fit classifier...'
        self.clf.fit(X, y)

    def predict(self, X):
        print 'predict classifier...'
        return self.clf.predict(X)
