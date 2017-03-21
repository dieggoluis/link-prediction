# coding=utf-8
# python 2.7
import numpy as np
import csv
import sys
import igraph
from sklearn.metrics import f1_score
from library import read_files, preprocessing
from sklearn.model_selection import ShuffleSplit

## global variable with the entire citation graph
network_graph = None

if __name__ == '__main__':

    # reading test and training set
    file_names = ["training_set.txt", "testing_set.txt"]
    files = read_files(file_names)
    training_set = files[0]
    testing_set = files[1]

    training_set = np.array([element[0].split(" ") for element in training_set])
    # this line reduces the training set to 5% of its size
    random_sample = np.random.choice(len(training_set), len(training_set)/20)
    training_set = training_set[random_sample]
    testing_set = np.array([element[0].split(" ") for element in testing_set])
    target_set = training_set[:,2]

    # clean tiles and abstracts
    # should be run only once (takes much time)
    if "--preprocessing" in sys.argv:
        # cleaning
        preprocessing()

    import feature_extractor
    import classifier
    
    ## cross validation
    if len(sys.argv) < 2 or sys.argv[1] == "--train":
        print "Cross validation..."
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
    
    ## training with entire training set
    elif sys.argv[1] == "--test": 
        print "Training with the entire training set..."
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
        print 'writing xgb_prediction_newfeatures.csv'
        with open("xgb_prediction_newfeatures.csv","wb") as pred1:
            csv_out = csv.writer(pred1)
            for row in predictions:
                csv_out.writerow(row)
    else:
        print "Invalid option."
        print "The options are:"
        print "    1. python user_test_submission.py --train"
        print "    2. python user_test_submission.py --test"
