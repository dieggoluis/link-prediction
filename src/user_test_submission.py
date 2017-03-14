# coding=utf-8
import numpy as np
import csv
from sklearn.metrics import f1_score
from library import read_files, preprocessing

from sklearn.model_selection import ShuffleSplit

if __name__ == '__main__':

    #preprocessing()

    import feature_extractor
    import classifier

    # clean 
    #   - text titles of node_information and save in cleaned_titles.csv
    #   - text abstracts of node_information and save in cleaned_abstracts.csv
    # should be run only once (takes much time)
    file_names = ["training_set.txt", "testing_set.txt"]
    files = read_files(file_names)
    training_set = files[0]
    testing_set = files[1]

    training_set = np.array([element[0].split(" ") for element in training_set])
    testing_set = np.array([element[0].split(" ") for element in testing_set])
    target_set = training_set[:,2]

    # cross validation
    '''
    skf = ShuffleSplit(n_splits=2, test_size=0.5, random_state=42)
    for train_is, test_is in skf.split(training_set, target_set):
        # training samples
        X_train = training_set[train_is][:,:-1].astype(int)
        y_train = target_set[train_is].astype(int)

        # Feature extraction
        fe = feature_extractor.FeatureExtractor()
        fe.fit(X_train, y_train)
        X_train = fe.transform(X_train)

        # test samples
        X_test = training_set[test_is][:,:-1].astype(int)
        y_test = target_set[test_is].astype(int)

        # Feature extraction
        X_test = fe.transform(X_test)

        # Classifier
        clf = classifier.Classifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # F1 score 
        print 'F1 score = ', f1_score(y_true=y_test, y_pred=y_pred)
    '''
    
    ## training with entire
    X_train = training_set[:][:,:-1].astype(int)
    y_train = target_set[:].astype(int)

    ## Feature extraction
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_train, y_train)
    X_train = fe.transform(X_train)

    ## test samples
    X_test = testing_set[:].astype(int)

    ## Feature extraction
    X_test = fe.transform(X_test)

    ## Classifier
    clf = classifier.Classifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    predictions = zip(range(len(y_pred)), y_pred)
    print 'writing xgb_predictions.csv'
    with open("xgb_predictions.csv","wb") as pred1:
        csv_out = csv.writer(pred1)
        for row in predictions:
            csv_out.writerow(row)
