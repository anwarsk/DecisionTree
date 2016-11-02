from DecisionTree.InputHandler import InputHandler
from BaggingBoosting import Constants
from BaggingBoosting.EnsembleUtil import EnsembleUtil

from pprint import pprint


class Main:

    def run(self):
        inputHandler = InputHandler()
        ensembleUtil = EnsembleUtil()

        # create data matrix from training file
        matrix_train = inputHandler.readFile(Constants.INPUT_FILE_NAME,
                                             Constants.LABEL_INDEX,
                                             Constants.FEATURE_INDICES,
                                             fileType="csv",
                                             ignoreHeader=True)
        pprint(matrix_train)

        bootstrap_matrix = ensembleUtil.createBootstrapSamples(matrix_train, Constants.NUMBER_OF_BAGS)
        pprint(bootstrap_matrix)

# run main function
if __name__ == '__main__':
    main_obj = Main()

    # dry run
    main_obj.run()