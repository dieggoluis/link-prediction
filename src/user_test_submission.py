import numpy as np
import csv

import feature_extractor
import classifier
from library import read_files

from sklearn.model_selection import ShuffleSplit

if __name__ == '__main__':

    file_names = ["training_set.txt", "testing_set.txt"]
    files = read_files(file_names)
    training_set = files[0]
    testing_set = files[1]

    training_set = np.array([element[0].split(" ") for element in training_set])
    testing_set = np.array([element[0].split(" ") for element in testing_set])
    target_set = training_set[:,2]

    # cross validation
    skf = ShuffleSplit(n_splits=2, test_size=0.5, random_state=42)
    for train_is, test_is in skf.split(training_set, target_set):
        # training samples
        X_train = training_set[train_is]
        y_train = target_set[train_is]

        # Feature extraction
        #fe = feature_extractor.FeatureExtractor()
        #fe.fit(X_train, y_train)
        #X_train = fe.transform(X_train)

        # test samples
        X_test = training_set[test_is]
        y_test = target_set[test_is]

        # Feature extraction
        #X_test = fe.transform(X_test)

        # Classifier
        #clf = classifier.Classifier()
        #clf.fit(X_train, y_train)
        #y_pred = clf.predict(X_test)

        # F1 score 
        #print('F1 score = ', f1_score(y_true=y_test, y_pred=y_pred)
