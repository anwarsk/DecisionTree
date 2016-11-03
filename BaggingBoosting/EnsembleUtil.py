import random
from DecisionTree.DecisionTree.Data import Data
from DecisionTree.DecisionTree.DecisionTree import DecisionTree


class EnsembleUtil:
    def createBootstrapSamples(self, trainingSet, numBags):
        bootstrap_samples = []
        sample_size = len(trainingSet)

        for index in range(numBags):
            print("Creating bootstrap sample #" + str(index))
            bootstrap_samples.append(
                [random.choice(trainingSet) for _ in range(sample_size)]
            )

        return bootstrap_samples

    def get_majority_voted_labels(self, predicted_label_collection):
        num_bags = len(predicted_label_collection)
        num_test_points = len(predicted_label_collection[0])

        # labels selected via majority vote - appearing more times
        majority_voted_labels = []

        for test_data_index in range(num_test_points):
            # keep count of majority vote
            label_count = {True.__int__(): 0,
                           False.__int__(): 0}

            # scan through all bags & increase vote
            for bag_index in range(num_bags):
                label_count[
                    predicted_label_collection[bag_index][test_data_index]
                ] += 1

            # add majority voted class-label
            majority_voted_labels.append(
                (label_count[True.__int__()] >= label_count[False.__int__()]).__int__()
            )

        # return majority vote
        return majority_voted_labels

    def calculate_accuracy(self, testing_data, predicted_classes):
        decision_tree = DecisionTree()

        # calculate accuracy of model
        dt_accuracy, dt_misclassification = decision_tree.calculateAccuracy(testing_data,
                                                                            predicted_classes)

        print('Accuracy of bagging ensemble = {}'.format(dt_accuracy))
        print('Misclassification Count = {}'.format(dt_misclassification))
        return


    def print_confusion_matrix(self, testing_data, predicted_classes):
        decision_tree = DecisionTree()

        # print confusion matrix
        decision_tree.plotConfusionMatrix(testing_data,
                                          predicted_classes)
        return
