#! /usr/bin/python

'''
This is a template outlining the functions we are expecting for us to be able to
interface with an call your code. This is not all of the functions you need. You
will also need to make sure to bring your decision tree learner in somehow
either by copying your code into a learn_decision_tree function or by importing
your decision tree code in from the file your wrote for PA#1. You will also need
some functions to handle creating your data bags, computing errors on your tree,
and handling the reweighting of data points.

For building your bags numpy's random module will be helpful.
'''

# This is the only non-native library to python you need
# import numpy as np;

from DecisionTree.DecisionTree.InputHandler import InputHandler
from DecisionTree.BaggingBoosting import Constants
from DecisionTree.BaggingBoosting.Main import Main

import sys, os;

# text formatting settings
BOLD = '\033[1m'
END = '\033[0m'

'''
Function: load_and_split_data(datapath)
datapath: (String) the location of the UCI mushroom data set directory in memory

This function loads the data set. datapath points to a directory holding
agaricuslepiotatest1.csv and agaricuslepiotatrain1.csv. The data from each file
is loaded and returned. All attribute values are nomimal. 30% of the data points
are missing a value for attribute 11 and instead have a value of "?". For the
purpose of these models, all attributes and data points are retained. The "?"
value is treated as its own attribute value.

Two nested lists are returned. The first list represents the training set and
the second list represents the test set.
'''


def load_data(datapath):
    inputHandler = InputHandler()

    # create data matrix from training file
    matrix_train = inputHandler.readFile(datapath + os.path.sep + get_train_filename(),
                                         Constants.LABEL_INDEX,
                                         fileType="csv",
                                         ignoreHeader=True)

    # create data matrix from testing file
    matrix_test = inputHandler.readFile(datapath + os.path.sep + get_test_filename(),
                                        Constants.LABEL_INDEX,
                                        fileType="csv",
                                        ignoreHeader=True)

    print("Number of training examples: " + str(matrix_train.__len__()))
    print("Number of testing examples: " + str(matrix_test.__len__()))

    # return train & test set
    return matrix_train, matrix_test


'''
Function: learn_bagged(tdepth, numbags, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numbags: (Integer)the number of bags to use to learn the trees
datapath: (String) the location in memory where the data set is stored

This function will manage coordinating the learning of the bagged ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''


def learn_bagged(tdepth, numbags, datapath):
    print(BOLD + '{0:{fill}{align}{width}} '.format('=',
                                                    fill='=',
                                                    width=65,
                                                    align='^') + END)
    print(BOLD + "BAGGING , CONFIGURATION : DEPTH = {}, NUM-BAGS = {}".format(tdepth,
                                                                              numbags) + END)
    print(BOLD + '{0:{fill}{align}{width}} '.format('=',
                                                    fill='=',
                                                    width=65,
                                                    align='^') + END)
    # get the train,test data sets
    train_data, test_data = load_data(datapath)

    # create instance of BaggingBoosting.Main class
    # this class contains logic for bagging ensemble method
    ensemble_main = Main(matrix_train=train_data,
                         matrix_test=test_data,
                         tree_depth=tdepth,
                         num_bags=numbags)

    # run the bagging algorithm
    ensemble_main.run_bagging()
    return;


'''
Function: learn_boosted(tdepth, numtrees, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numtrees: (Integer) the number of boosted trees to learn
datapath: (String) the location in memory where the data set is stored

This function wil manage coordinating the learning of the boosted ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''


def learn_boosted(tdepth, numtrees, datapath):
    print(BOLD + '{0:{fill}{align}{width}} '.format('=',
                                                    fill='=',
                                                    width=65,
                                                    align='^') + END)
    print(BOLD + "ADABOOST , CONFIGURATION : DEPTH = {}, NUM-TREES = {}".format(tdepth,
                                                                               numtrees) + END)
    print(BOLD + '{0:{fill}{align}{width}} '.format('=',
                                                    fill='=',
                                                    width=65,
                                                    align='^') + END)
    # get the train,test data sets
    train_data, test_data = load_data(datapath)

    ensemble_main = Main(matrix_train=train_data,
                         matrix_test=test_data,
                         tree_depth=tdepth,
                         num_bags=numtrees)

    # run the bagging algorithm
    ensemble_main.run_boosting()
    return;


'''
Function: get_train_filename()

This function returns the name of the training data file
named agaricuslepiotatrain1.csv, which is located in the
mushrooms directory in memory
'''


def get_train_filename():
    return 'agaricuslepiotatrain1.csv'


'''
Function: get_test_filename()

This function returns the name of the testing data file
named agaricuslepiotatest1.csv, which is located in the
mushrooms directory in memory
'''


def get_test_filename():
    return 'agaricuslepiotatest1.csv'


if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <ensemble_type> <tree_depth> <num_bags/trees> <data_set_path>
    # Ex. bag 3 10 mushrooms
    # Ex. boost 1 10 mushrooms

    # Get the ensemble type
    entype = sys.argv[1];
    # Get the depth of the trees
    tdepth = int(sys.argv[2]);
    # Get the number of bags or trees
    nummodels = int(sys.argv[3]);
    # Get the location of the data set
    datapath = sys.argv[4];

    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bagged decision tree ensemble
        learn_bagged(tdepth, nummodels, datapath);
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath);
