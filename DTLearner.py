# Author: Sophie Zhao

import math
import sys
import numpy as np
import logging

logging.basicConfig(filename="DTLearner_log.log", level=logging.DEBUG, format='%(asctime)s:%(levelname)s %(message)s')


class DTLearner():
    def __init__(self, leaf_size = 1, verbose):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.dtree = None

    def get_highest_correlation(self, Xs, Y):
        # Find the column as split factor: the one with the highest correlation
        #
        # Parameters:
        # Xs (numpy.ndarray) – A set of feature values used to train the learner
        # Y (numpy.ndarray) – The value we are attempting to predict given the X data
        highest_correlation = -1
        highest_correlation_column = 0

        # go through data in all columns
        for i in range(Xs.shape[1]):
            if np.std(Xs[:, i]) == 0:
                curr_correlation = 0
            else:
                curr_correlation = abs(np.corrcoef(Xs[:, i], Y)[
                                           0, 1])  # [0, 1]: the correlation coefficient between the 1st variable (Xs[:, i]) and the 2nd variable (Y).
            if curr_correlation > highest_correlation:
                highest_correlation = curr_correlation
                highest_correlation_column = i
        return int(highest_correlation_column)

    def node(self, factor_used, split_val, left, right):
        dt_row = np.asarray([factor_used, split_val, left, right])
        return dt_row

    def build_tree(self, Xs, Y):
        # Build binary decision tree from training data
        #
        # Parameters:
        # Xs  – X factors
        # Y – Y factors

        if self.verbose:
            logging.debug("Build_tree: Xs:  ")
            logging.debug(Xs)
            logging.debug("Build_tree: Y:  ")
            logging.debug(Y)

        # base case:
        if Xs.shape[0] <= self.leaf_size:
            return self.node(np.nan, np.mean(Y), np.nan, np.nan)

        elif np.all(np.isclose(Y, Y[0])):
            return self.node(np.nan, Y[0], np.nan, np.nan)

        elif np.all(Y == Y[0]):
            return self.node(np.nan, Y[0], np.nan, np.nan)

        # find the highest correlation among all columns
        decision_factor_index = self.get_highest_correlation(Xs, Y)

        if self.verbose:
            logging.debug("decision_factor_index: ")
            logging.debug(decision_factor_index)

        # find median as the split value, then slip the dataset
        decision_factor_median = np.median(Xs[:, decision_factor_index])
        data_split_mask = Xs[:, decision_factor_index] <= decision_factor_median

        # When all values in data_split_mask are True, make half false
        if np.all(data_split_mask):
            data_split_mask[int(data_split_mask.shape[0]/2):] = False

        left_half_Xs = Xs[data_split_mask]
        right_half_Xs = Xs[~data_split_mask]
        left_half_Y = Y[data_split_mask]
        right_half_Y = Y[~data_split_mask]

        # build left tree
        left_tree = self.build_tree(left_half_Xs, left_half_Y)
        if self.verbose:
            logging.debug("building left tree: ")
            logging.debug(left_tree)

        # build right tree
        if self.verbose:
            logging.debug("----start building right tree---: ")

        right_tree = self.build_tree(right_half_Xs, right_half_Y)
        if self.verbose:
            logging.debug("building right tree: ")
            logging.debug(right_tree)

        if left_tree.ndim != 1:
            root = self.node(int(decision_factor_index), decision_factor_median, 1, 1 + left_tree.shape[0])
            if self.verbose:
                logging.debug("building root !=1 : ")
                logging.debug(root)
        else:  # if the number of dimension is one, meaning left tree is a leaf
            root = self.node(int(decision_factor_index), decision_factor_median, 1, 2)
            if self.verbose:
                logging.debug("building root = 1: ")
                logging.debug(root)

        root = np.row_stack((root, left_tree, right_tree))
        if self.verbose:
            logging.debug("stacked root: ")
            logging.debug(root)

        return root


    def add_evidence(self, x_train, y_train):
        # Add training data to learner
        #
        # Parameters:
        # data_x (numpy.ndarray) – A set of feature values used to train the learner
        # data_y (numpy.ndarray) – The value we are attempting to predict given the X data
        logging.debug("------x_train-----")
        logging.debug(x_train)
        logging.debug("------y_train-----")
        logging.debug(y_train)

        self.dtree = self.build_tree(x_train, y_train)
        if self.verbose:
            logging.debug("------dtree-----")
            logging.debug(self.dtree)

    def query_helper(self, one_query):
        root_index = 0
        while not np.isnan(self.dtree[root_index][0]):
            root = self.dtree[int(root_index)]
            factor_index = int(root[0])
            split_value = root[1]
            query_value = one_query[factor_index]
            if query_value <= split_value:  # goes to the left
                left = int(root[2])
                root_index += left
            else:  # goes to the right
                right = int(root[3])
                root_index += right

        result = self.dtree[root_index][1]

        return result


    def query(self, points):
        # Estimate a set of test points given the model we built.
        #
        # Parameters:
        # points (numpy.ndarray) – A numpy array with each row corresponding to a specific query.
        # Returns:
        # The predicted result of the input data according to the trained model
        # Return type:
        # numpy.ndarray

        # lambda function curr_query, which wraps the query_helper method, making it compatible with numpy.apply_along_axis()
        curr_query = lambda one_query: self.query_helper(one_query)
        # iterates over each row of points and applies the query_helper method to that row
        result = np.apply_along_axis(curr_query, 1, points)
        return result
