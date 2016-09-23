## This file holds the constants and input parameters to the program

# import for drawing tree (optional)
import pydot

# Input file to train the data
INPUT_FILE_TRAIN = "/Users/goshenoy/SOIC-Courses/Applied-ML/PA1/monks-1.train.txt"
INPUT_FILE_TEST = "/Users/goshenoy/SOIC-Courses/Applied-ML/PA1/monks-1.test.txt"

LABEL_INDEX = 1
FEATURE_INDICES = (2,8)
TREE_DEPTH = 10

GRAPH = pydot.Dot(graph_type='graph')