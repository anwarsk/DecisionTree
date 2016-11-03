from DecisionTree.DecisionTree.InputHandler import InputHandler
from DecisionTree.BaggingBoosting import Constants
from DecisionTree.BaggingBoosting.BaggingEnsemble import BaggingEnsemble
from DecisionTree.BaggingBoosting.EnsembleUtil import EnsembleUtil

from pprint import pprint


class Main:

    # text formatting settings
    BOLD = '\033[1m'
    END = '\033[0m'

    def run_bagging(self):
        inputHandler = InputHandler()
        ensemble_util = EnsembleUtil()

        # create data matrix from training file
        matrix_train = inputHandler.readFile(Constants.TRAIN_FILE_NAME,
                                             Constants.LABEL_INDEX,
                                             fileType="csv",
                                             ignoreHeader=True)

        # create data matrix from testing file
        matrix_test = inputHandler.readFile(Constants.TEST_FILE_NAME,
                                            Constants.LABEL_INDEX,
                                            fileType="csv",
                                            ignoreHeader=True)

        print("Number of training examples: " + str(matrix_train.__len__()))
        print("Number of testing examples: " + str(matrix_test.__len__()))

        # initialize bagging ensemble
        bagging_ensemble = BaggingEnsemble(training_data=matrix_train,
                                           num_bags=Constants.NUMBER_OF_BAGS)

        # learn the bagging ensemble
        bagging_ensemble.learn(Constants.TREE_DEPTH)

        # predict using bagging
        predicted_classes = bagging_ensemble.predict(testing_data=matrix_test)
        # print(predicted_classes)

        # print accuracy & misclassification count
        print("\n" + Main.BOLD + "TASK: Print Bagging accuracy & mis-classification count" + Main.END)
        ensemble_util.calculate_accuracy(testing_data=matrix_test,
                                         predicted_classes=predicted_classes)

        # print confusion matrix
        print("\n" + Main.BOLD + "TASK: Print Confusion Matrix" + Main.END)
        ensemble_util.print_confusion_matrix(testing_data=matrix_test,
                                             predicted_classes=predicted_classes)

        return


# run main function
if __name__ == '__main__':
    main_obj = Main()

    # dry run
    main_obj.run_bagging()
