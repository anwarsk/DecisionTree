## This file holds the constants and input parameters to the program

# Path format of monk input data files to the program
# INPUT_FILE_NAME = "/home/anwar/AML-PA-1/raw_data/monks-{index}.{purpose}"
INPUT_FILE_NAME = "/Users/goshenoy/SOIC-Courses/Applied-ML/Programming_Assignments/PA2/mushrooms/agaricuslepiotatrain1.csv"

# Output result file from the program
RESULT_FILE = "dt_accuracies.csv"

# Index of the class label in monk data
LABEL_INDEX = 20

# Indices of features in the monk data
FEATURE_INDICES = (0,20)

# Desired max depth of the tree
TREE_DEPTH = 3

# Desired number of sample bags
NUMBER_OF_BAGS = 2