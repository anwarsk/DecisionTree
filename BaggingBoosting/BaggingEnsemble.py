from DecisionTree.BaggingBoosting.EnsembleUtil import EnsembleUtil
from DecisionTree.DecisionTree.Data import Data
from DecisionTree.DecisionTree.DecisionTree import DecisionTree

from pprint import pprint


class BaggingEnsemble:

    # text formatting settings
    BOLD = '\033[1m'
    END = '\033[0m'

    def __init__(self, training_data, num_bags):
        self.__training_data = training_data
        self.__num_bags = num_bags
        self.__bootstrap_samples = []
        self.__learned_models = []
        self.__predicted_classes_collection = []


    def learn(self, tree_depth):
        ensembleUtil = EnsembleUtil()

        # get bootstrap samples
        print("\n" + BaggingEnsemble.BOLD + "TASK: Create Bootstrap Samples from Training Data" + BaggingEnsemble.END)
        self.__bootstrap_samples = ensembleUtil.createBootstrapSamples(self.__training_data,
                                                                       self.__num_bags)

        print("\n" + BaggingEnsemble.BOLD + "TASK: Learn Decision Tree for each Bootstrap Sample" + BaggingEnsemble.END)
        # for each bootstrap sample, learn a DT
        for count, bootstrap_sample in enumerate(self.__bootstrap_samples):
            print("Learning DT for bootstrapped sample #" + str(count))
            data_train = Data()
            decision_tree = DecisionTree()

            # run the decision-tree training algorithm
            data_train.setMatrix(bootstrap_sample)
            decision_tree.train(data=data_train,
                                treeDepth=tree_depth)

            # save the learned model
            self.__learned_models.append(decision_tree)
        return


    def predict(self, testing_data):
        ensembleUtil = EnsembleUtil()

        # predict using each DT model
        print("\n" + BaggingEnsemble.BOLD + "TASK: Get predicted class label by each learned DT model" + BaggingEnsemble.END)

        # create test data matrix
        data_test = Data()
        data_test.setMatrix(testing_data)

        for count, dt_model in enumerate(self.__learned_models):
            print("Predicting class labels with DT model #" + str(count))
            self.__predicted_classes_collection.append(
                dt_model.test(data_test.getMatrix())
            )

        #print(self.__predicted_classes_collection)
        return ensembleUtil.get_majority_voted_labels(self.__predicted_classes_collection)