from DecisionTree.DecisionTree.InputHandler import InputHandler
from DecisionTree.BaggingBoosting import Constants
from DecisionTree.BaggingBoosting.BaggingLearner import BaggingLearner

from pprint import pprint


class Main:

    def run(self):
        inputHandler = InputHandler()

        # create data matrix from training file
        matrix_train = inputHandler.readFile(Constants.INPUT_FILE_NAME,
                                             Constants.LABEL_INDEX,
                                             fileType="csv",
                                             ignoreHeader=True)
        print("Number of training examples: " + str(matrix_train.__len__()))

        bagging_learner = BaggingLearner(training_data=matrix_train,
                                         num_bags=Constants.NUMBER_OF_BAGS)

        bagging_learner.learn(Constants.TREE_DEPTH)


# run main function
if __name__ == '__main__':
    main_obj = Main()

    # dry run
    main_obj.run()