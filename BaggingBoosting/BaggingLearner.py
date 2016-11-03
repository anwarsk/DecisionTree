from DecisionTree.BaggingBoosting.EnsembleUtil import EnsembleUtil
from DecisionTree.DecisionTree.Data import Data
from DecisionTree.DecisionTree.DecisionTree import DecisionTree

from pprint import pprint


class BaggingLearner:

    def __init__(self, training_data, num_bags):
        self.__training_data = training_data
        self.__num_bags = num_bags
        self.__bootstrap_samples = []
        self.__learned_models = []


    def learn(self, tree_depth):
        ensembleUtil = EnsembleUtil()

        # get bootstrap samples
        print("\nStep 1: Create Bootstrap Samples from Training Data")
        self.__bootstrap_samples = ensembleUtil.createBootstrapSamples(self.__training_data,
                                                                       self.__num_bags)
        print("\nStep 2: Learn Decision Tree for each Bootstrap Sample")
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
