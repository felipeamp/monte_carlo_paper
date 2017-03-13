#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing the DecisionTree, TreeNode and NodeSplit classes.
"""

import math
import random
import sys
import timeit

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import chi2

import monte_carlo


MIN_ALLOWED_IN_TWO_LARGEST = 40
MAX_P_VALUE_CHI_SQUARE_TEST = 0.1


class DecisionTree(object):
    """Data structure containing basic information pertaining to the whole tree.

        Every attribute is protected. This class' state should be accessed only indirectly, through
    its methods.

    Args:
        criterion (Criterion): criterion which will be used to generate the tree nodes/splits.
        is_monte_carlo_criterion (bool, optional): indicates if the splitting criterion uses our
            Monte Carlo framework. Defaults to `False`.
        upper_p_value_threshold (float, optional): the p-value-upper-threshold for our Monte Carlo
            framework. If an attribute has a p-value above this threshold, it will be rejected with
            probability `prob_monte_carlo`. Defaults to `None`.
        lower_p_value_threshold (float, optional): the p-value-lower-threshold for our Monte Carlo
            framework. If an attribute has a p-value below this threshold, it will be accepted with
            probability `prob_monte_carlo`. Defaults to `None`.
        prob_monte_carlo (float, optional): the probability of accepting an attribute with p-value
            smaller than `lower_p_value_threshold` and rejecting an attribute with p-value greater
            than `upper_p_value_threshold` for our Monte Carlo framework. Defaults to `None`.
    """
    def __init__(self, criterion, is_monte_carlo_criterion=False, upper_p_value_threshold=None,
                 lower_p_value_threshold=None, prob_monte_carlo=None):
        #TESTED!
        self._criterion = criterion
        self._dataset = None
        self._root_node = None
        self._is_monte_carlo_criterion = is_monte_carlo_criterion
        self._upper_p_value_threshold = upper_p_value_threshold
        self._lower_p_value_threshold = lower_p_value_threshold
        self._prob_monte_carlo = prob_monte_carlo

    def get_root_node(self):
        """Returns the TreeNode at the root of the tree. Might be None.
        """
        return self._root_node

    def get_tree_time_num_tests_fails(self):
        '''Returns the total time taken to calculate the number of tests and fails allowed at each
        node in the tree.'''
        return self._root_node.get_subtree_time_num_tests_fails()

    def get_tree_time_expected_tests(self):
        '''Returns the total time taken to calculate the total expected number of tests at each node
        in the tree.'''
        return self._root_node.get_subtree_time_expected_tests()

    def _classify_sample(self, sample, sample_key):
        #TESTED!
        if self._root_node is None:
            print('Cannot classify in untrained tree!')
            sys.exit(1)
        curr_node = self._root_node
        classified_with_unkown_value = False
        unkown_value_attrib_index = None
        while not curr_node.is_leaf:
            split_attrib_index = curr_node.node_split.separation_attrib_index
            sample_value = sample[split_attrib_index]
            if self._dataset.valid_numeric_attribute[split_attrib_index]:
                if sample_value is None:
                    print('\tSample {} has value unkown to split'
                          ' (value = {} in attrib #{}).'.format(
                              sample_key,
                              sample_value,
                              curr_node.node_split.separation_attrib_index))
                    classified_with_unkown_value = True
                    unkown_value_attrib_index = curr_node.node_split.separation_attrib_index
                    break
                mid_point = curr_node.node_split.mid_point
                if sample_value <= mid_point:
                    curr_node = curr_node.nodes[0]
                else:
                    curr_node = curr_node.nodes[1]
            else:
                try:
                    split_index = curr_node.node_split.values_to_split[sample_value]
                    curr_node = curr_node.nodes[split_index]
                except KeyError:
                    print('\tSample {} has value unkown to split'
                          ' (value = {} in attrib #{}).'.format(
                              sample_key,
                              sample_value,
                              curr_node.node_split.separation_attrib_index))
                    classified_with_unkown_value = True
                    unkown_value_attrib_index = curr_node.node_split.separation_attrib_index
                    break
        return (curr_node.most_common_int_class,
                classified_with_unkown_value,
                unkown_value_attrib_index)

    def _classify_samples(self, test_dataset_sample, test_dataset_sample_class,
                          test_dataset_sample_costs, test_samples_indices,
                          test_dataset_sample_keys):
        print('Starting classifications...')
        classifications = []
        classified_with_unkown_value_array = []
        unkown_value_attrib_index_array = []

        num_correct_classifications = 0
        num_correct_classifications_wo_unkown = 0
        num_unkown = 0

        total_cost = 0.0
        total_cost_wo_unkown = 0.0

        for test_sample_index in test_samples_indices:
            (predicted_class,
             classified_with_unkown_value,
             unkown_value_attrib_index) = self._classify_sample(
                 test_dataset_sample[test_sample_index],
                 test_dataset_sample_keys[test_sample_index])
            classifications.append(predicted_class)
            classified_with_unkown_value_array.append(classified_with_unkown_value)
            unkown_value_attrib_index_array.append(unkown_value_attrib_index)
            if predicted_class == test_dataset_sample_class[test_sample_index]:
                num_correct_classifications += 1
                if not classified_with_unkown_value:
                    num_correct_classifications_wo_unkown += 1
            total_cost += test_dataset_sample_costs[test_sample_index][predicted_class]
            if not classified_with_unkown_value:
                total_cost_wo_unkown += test_dataset_sample_costs[
                    test_sample_index][predicted_class]
            else:
                num_unkown += 1
        print('Done!')
        return (classifications,
                num_correct_classifications,
                num_correct_classifications_wo_unkown,
                total_cost,
                total_cost_wo_unkown,
                classified_with_unkown_value_array,
                num_unkown,
                unkown_value_attrib_index_array)

    def train(self, dataset, training_samples_indices, max_depth, min_samples_per_node,
              max_p_value=None, use_stop_conditions=False):
        """Trains the tree in a recursive fashion, starting at the root's TreeNode.

        Arguments:
            dataset (Dataset): dataset containing the samples used for training.
            training_samples_indices (:obj:'list' of 'int'): list containing the indices of samples
                of `dataset` used for training.
            max_depth (int): maximum tree depth allowed. Zero means the root is a leaf.
            min_samples_per_node (int): if a node has less than this number of training samples, it
                will necessarily be a leaf.
            max_p_value (float, optional): only used for some max cut criteria. It is the maximum
                p-value allowed for a split. Defaults to `None`.
            use_stop_conditions (bool, optional): informs wether we should use prunning techniques
                to avoid using attributes with small number of samples (and, thus, avoiding
                statistical anomalies). An attribute will be considered invalid if it contains less
                than `MIN_ALLOWED_IN_TWO_LARGEST` samples in the second largest class (this way at
                least two classes have this number of samples) or if a chi-square test, applied on
                the attributes' contingency table has a p-value greater or equal to
                `MAX_P_VALUE_CHI_SQUARE_TEST`. When an attribute is considered invalid for the above
                reasons, this information will be passed to every child node of the current
                TreeNode. Note that numeric attributes are never tested in this way.Defaults to
                `False`.
        """
        #TESTED!
        self._dataset = dataset
        print('Starting tree training...')
        self._root_node = TreeNode(dataset,
                                   training_samples_indices,
                                   dataset.valid_nominal_attribute[:],
                                   max_depth,
                                   min_samples_per_node,
                                   use_stop_conditions,
                                   is_monte_carlo_criterion=self._is_monte_carlo_criterion,
                                   upper_p_value_threshold=self._upper_p_value_threshold,
                                   lower_p_value_threshold=self._lower_p_value_threshold,
                                   prob_monte_carlo=self._prob_monte_carlo)
        self._root_node.create_subtree(self._criterion, max_p_value)
        print('Done!')

    def train_and_self_validate(self, dataset, training_samples_indices,
                                validation_sample_indices, max_depth, min_samples_per_node,
                                max_p_value=None, use_stop_conditions=False):
        """Trains a tree with part of the dataset (training samples) and tests the tree
        classification in another part (validation samples).

        Note that although the training and test samples are part of the same Dataset class, they
        usually shouldn't intersect.

        Arguments:
            dataset (Dataset): dataset containing the samples used for training.
            training_samples_indices (:obj:'list' of 'int'): list containing the indices of samples
                of `dataset` used for training.
            validation_sample_indices (:obj:'list' of 'int'): list containing the indices of samples
                of `dataset` used to test the tree classification.
            max_depth (int): maximum tree depth allowed. Zero means the root is a leaf.
            min_samples_per_node (int): if a node has less than this number of training samples, it
                will necessarily be a leaf.
            max_p_value (float, optional): only used for some max cut criteria. It is the maximum
                p-value allowed for a split. Defaults to `None`.
            use_stop_conditions (bool, optional): informs wether we should use prunning techniques
                to avoid using attributes with small number of samples (and, thus, avoiding
                statistical anomalies). An attribute will be considered invalid if it contains less
                than `MIN_ALLOWED_IN_TWO_LARGEST` samples in the second largest class (this way at
                least two classes have this number of samples) or if a chi-square test, applied on
                the attributes' contingency table has a p-value greater or equal to
                `MAX_P_VALUE_CHI_SQUARE_TEST`. When an attribute is considered invalid for the above
                reasons, this information will be passed to every child node of the current
                TreeNode. Note that numeric attributes are never tested in this way.Defaults to
                `False`.

        Returns:
            A tuple containing the tree's max depth in the second entry and, in the first entry,
                another tuple. This (first) tuple contains, in order:
                - a list of predicted class for each validation sample;
                - the number of correct classifications;
                - the number of correct classifications done without validation samples with unkown
                    values (that is, values that are unkown at a TreeNode -- they are classified as
                    the most common class at that node);
                - the total cost of the classification errors (when errors costs are uniform, this
                    is equal to the total number of validation samples minus the number of correct
                    classifications);
                -  the total cost of the classification errors without considering validation
                    samples with unkown values;
                - a list of booleans indicating if the i-th validation sample was classified with an
                    unkown value;
                - the number of validation samples classified with unkown values;
                - list where the i-th entry has the attribute index used for classification of the
                    i-th sample when an unkown value occurred.
        """
        self.train(dataset,
                   training_samples_indices,
                   max_depth,
                   min_samples_per_node,
                   max_p_value,
                   use_stop_conditions)
        max_depth = self.get_root_node().get_max_depth()
        return (self._classify_samples(self._dataset.samples,
                                       self._dataset.sample_class,
                                       self._dataset.sample_costs,
                                       validation_sample_indices,
                                       self._dataset.sample_index_to_key),
                max_depth)

    def cross_validate(self, dataset, num_folds, max_depth, min_samples_per_node, max_p_value=None,
                       is_stratified=True, print_tree=False, seed=None, print_samples=False,
                       use_stop_conditions=False):
        """Does a cross-validation using a given dataset.

        It splits this dataset in `num_folds` folds and calls `train_and_self_validate` on each.
        Might be given a seed for the dataset's random splitting and might be stratified.

        Arguments:
            dataset (Dataset): dataset containing the samples used for training.
            num_folds (int): number of folds used in the cross-validation.
            max_depth (int): maximum tree depth allowed. Zero means the root is a leaf.
            min_samples_per_node (int): if a node has less than this number of training samples, it
                will necessarily be a leaf.
            max_p_value (float, optional): only used for some max cut criteria. It is the maximum
                p-value allowed for a split. Defaults to `None`.
            is_stratified (bool, optional): Indicates wheter the cross-validation should be
                stratified or just a simple k-fold cross-validation. Stratified means the samples'
                splitting will try to keep the classes' distribution the same across folds. Defaults
                to `True`.
            print_tree (bool, optional): Indicates wether the trees of every fold should be printed
                to stdout. Defaults to `False`.
            seed (int, optional): indicates the seed that should be used to generate the random
                samples' splitting in folds. If `None`, a random seed is used. Defaults to `None`.
            print_samples (bool, optional): if `True`, prints the samples indices used at each fold.
                Used for debugging. Defaults to `False`.
            use_stop_conditions (bool, optional): informs wether we should use prunning techniques
                to avoid using attributes with small number of samples (and, thus, avoiding
                statistical anomalies). An attribute will be considered invalid if it contains less
                than `MIN_ALLOWED_IN_TWO_LARGEST` samples in the second largest class (this way at
                least two classes have this number of samples) or if a chi-square test, applied on
                the attributes' contingency table has a p-value greater or equal to
                `MAX_P_VALUE_CHI_SQUARE_TEST`. When an attribute is considered invalid for the above
                reasons, this information will be passed to every child node of the current
                TreeNode. Note that numeric attributes are never tested in this way.Defaults to
                `False`.

        Returns:
            A tuple containing, in order:
                - a list of predicted class for each sample;
                - the number of correct classifications;
                - the number of correct classifications done without samples with unkown values;
                    (that is, values that are unkown at a TreeNode -- they are classified as the
                    most common class at that node);
                - the total cost of the classification errors (when errors costs are uniform, this
                    is equal to the total number of samples minus the number of correct
                    classifications);
                - the total cost of the classification errors without considering samples with
                    unkown values;
                - a list of booleans indicating if the i-th sample was classified with an unkown
                    value;
                - the number of samples classified with unkown values;
                - list where the i-th entry has the attribute index used for classification of the
                    i-th sample when an unkown value occurred;
                - list containing the nodes information (see TreeNode.get_nodes_infos()) for each
                    fold.
                - list containing the maximum tree depth for each fold.
        """

        classifications = [0] * dataset.num_samples
        num_correct_classifications = 0
        num_correct_classifications_wo_unkown = 0
        total_cost = 0.0
        total_cost_wo_unkown = 0.0
        classified_with_unkown_value_array = [False] * dataset.num_samples
        num_unkown = 0
        unkown_value_attrib_index_array = [0] * dataset.num_samples
        max_depth_per_fold = []

        fold_count = 0

        nodes_infos_per_fold = []

        sample_indices_and_classes = list(enumerate(dataset.sample_class))
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        random.shuffle(sample_indices_and_classes)
        shuffled_sample_indices, shuffled_sample_classes = zip(*sample_indices_and_classes)

        if is_stratified:
            for (training_randomized_indices,
                 validation_randomized_indices) in StratifiedKFold(n_splits=num_folds).split(
                     shuffled_sample_indices,
                     shuffled_sample_classes):

                training_samples_indices = [shuffled_sample_indices[index]
                                            for index in training_randomized_indices]
                validation_sample_indices = [shuffled_sample_indices[index]
                                             for index in validation_randomized_indices]

                if print_samples:
                    print('Samples used for validation in this fold:')
                    print(validation_sample_indices)
                    print()

                ((curr_classifications,
                  curr_num_correct_classifications,
                  curr_num_correct_classifications_wo_unkown,
                  curr_total_cost,
                  curr_total_cost_wo_unkown,
                  curr_classified_with_unkown_value_array,
                  curr_num_unkown,
                  curr_unkown_value_attrib_index_array),
                 curr_max_depth) = self.train_and_self_validate(
                     dataset,
                     training_samples_indices,
                     validation_sample_indices,
                     max_depth,
                     min_samples_per_node,
                     max_p_value,
                     use_stop_conditions)

                max_depth_per_fold.append(curr_max_depth)
                for curr_index, validation_sample_index in enumerate(validation_sample_indices):
                    classifications[validation_sample_index] = curr_classifications[curr_index]
                    classified_with_unkown_value_array[validation_sample_index] = (
                        curr_classified_with_unkown_value_array[curr_index])
                    unkown_value_attrib_index_array[validation_sample_index] = (
                        curr_unkown_value_attrib_index_array[curr_index])
                num_correct_classifications += curr_num_correct_classifications
                num_correct_classifications_wo_unkown += curr_num_correct_classifications_wo_unkown
                total_cost += curr_total_cost
                total_cost_wo_unkown += curr_total_cost_wo_unkown
                num_unkown += curr_num_unkown

                fold_count += 1
                nodes_infos_per_fold.append(self._root_node.get_nodes_infos())

                if print_tree:
                    print()
                    print('-' * 50)
                    print('Fold:', fold_count)
                    self.save_tree()
        else:
            for (training_samples_indices,
                 validation_sample_indices) in KFold(n_splits=num_folds).split(
                     shuffled_sample_indices):

                ((curr_classifications,
                  curr_num_correct_classifications,
                  curr_num_correct_classifications_wo_unkown,
                  curr_total_cost,
                  curr_total_cost_wo_unkown,
                  curr_classified_with_unkown_value_array,
                  curr_num_unkown,
                  curr_unkown_value_attrib_index_array),
                 curr_max_depth) = self.train_and_self_validate(
                     dataset,
                     training_samples_indices,
                     validation_sample_indices,
                     max_depth,
                     min_samples_per_node,
                     max_p_value,
                     use_stop_conditions)

                max_depth_per_fold.append(curr_max_depth)
                for curr_index, validation_sample_index in enumerate(validation_sample_indices):
                    classifications[validation_sample_index] = curr_classifications[curr_index]
                    classified_with_unkown_value_array[validation_sample_index] = (
                        curr_classified_with_unkown_value_array[curr_index])
                    unkown_value_attrib_index_array[validation_sample_index] = (
                        curr_unkown_value_attrib_index_array[curr_index])
                num_correct_classifications += curr_num_correct_classifications
                num_correct_classifications_wo_unkown += curr_num_correct_classifications_wo_unkown
                total_cost += curr_total_cost
                total_cost_wo_unkown += curr_total_cost_wo_unkown
                num_unkown += curr_num_unkown

                fold_count += 1
                nodes_infos_per_fold.append(self._root_node.get_nodes_infos())

                if print_tree:
                    print()
                    print('-' * 50)
                    print('Fold:', fold_count)
                    self.save_tree()

        return  (classifications, num_correct_classifications,
                 num_correct_classifications_wo_unkown, total_cost, total_cost_wo_unkown,
                 classified_with_unkown_value_array, num_unkown, unkown_value_attrib_index_array,
                 nodes_infos_per_fold, max_depth_per_fold)

    def test(self, test_sample_indices):
        """Tests the (already trained) tree over samples from the same dataset as the
            training set. If the tree hasn't been trained, the program will exit.

        Arguments:
            test_sample_indices (:obj:'list' of 'int'): list of the test set indices for samples
                from the same dataset used for training.

        Returns:
            A tuple containing, in order:
                - a list of predicted class for each test sample;
                - the number of correct classifications;
                - the number of correct classifications done without test samples with unkown
                    values (that is, values that are unkown at a TreeNode -- they are classified as
                    the most common class at that node);
                - the total cost of the classification errors (when errors costs are uniform, this
                    is equal to the total number of test samples minus the number of correct
                    classifications);
                -  the total cost of the classification errors without considering test
                    samples with unkown values;
                - a list of booleans indicating if the i-th test sample was classified with an
                    unkown value;
                - the number of test samples classified with unkown values;
                - list where the i-th entry has the attribute index used for classification of the
                    i-th sample when an unkown value occurred.
        """

        if self._root_node is None:
            print('Decision tree must be trained before testing.')
            sys.exit(1)
        return self._classify_samples(self._dataset.test_samples,
                                      self._dataset.test_sample_class,
                                      self._dataset.test_sample_costs,
                                      test_sample_indices,
                                      self._dataset.test_sample_index_to_key)

    def test_from_csv(self, test_dataset_csv_filepath, key_attrib_index, class_attrib_index,
                      split_char, missing_value_string):
        """Tests the (already trained) tree using all samples from a given csv file. If the tree
        hasn't been trained, the program will exit.

        Arguments:
            test_dataset_csv_filepath (str): path to the test dataset.
            key_attrib_index (int): column index of the samples' keys on the csv.
            class_attrib_index (int): column index of the samples' classes on the csv.
            split_char (str): char used to split columns in the csv.
            missing_value_string (str): string used to indicate that a sample does not have a value.

        Returns:
            A tuple containing, in order:
                - a list of predicted class for each test sample;
                - the number of correct classifications;
                - the number of correct classifications done without test samples with unkown
                    values (that is, values that are unkown at a TreeNode -- they are classified as
                    the most common class at that node);
                - the total cost of the classification errors (when errors costs are uniform, this
                    is equal to the total number of test samples minus the number of correct
                    classifications);
                -  the total cost of the classification errors without considering test
                    samples with unkown values;
                - a list of booleans indicating if the i-th test sample was classified with an
                    unkown value;
                - the number of test samples classified with unkown values;
                - list where the i-th entry has the attribute index used for classification of the
                    i-th sample when an unkown value occurred.
        """

        if self._root_node is None or self._dataset is None:
            print('Decision tree must be trained before testing.')
            sys.exit(1)
        self._dataset.load_test_set_from_csv(test_dataset_csv_filepath,
                                             key_attrib_index,
                                             class_attrib_index,
                                             split_char,
                                             missing_value_string)
        return self._classify_samples(self._dataset.test_samples,
                                      self._dataset.test_sample_class,
                                      self._dataset.test_sample_costs,
                                      list(range(len(self._dataset.test_sample_index_to_key))),
                                      self._dataset.test_sample_index_to_key)

    def train_and_test(self, dataset, training_samples_indices, test_sample_indices,
                       max_depth, min_samples_per_node, max_p_value=None,
                       use_stop_conditions=False):
        """Trains a tree with part of the dataset (training samples) and tests the tree
        classification in another part (validation samples).

        Note that although the training and test samples are part of the same Dataset class, they
        usually shouldn't intersect.

        Arguments:
            dataset (Dataset): dataset containing the samples used for training.
            training_samples_indices (:obj:'list' of 'int'): list containing the indices of samples
                of `dataset` used for training.
            test_sample_indices (:obj:'list' of 'int'): list containing the indices of samples of
                `dataset` used to test the tree classification.
            max_depth (int): maximum tree depth allowed. Zero means the root is a leaf.
            min_samples_per_node (int): if a node has less than this number of training samples, it
                will necessarily be a leaf.
            max_p_value (float, optional): only used for some max cut criteria. It is the maximum
                p-value allowed for a split. Defaults to `None`.
            use_stop_conditions (bool, optional): informs wether we should use prunning techniques
                to avoid using attributes with small number of samples (and, thus, avoiding
                statistical anomalies). An attribute will be considered invalid if it contains less
                than `MIN_ALLOWED_IN_TWO_LARGEST` samples in the second largest class (this way at
                least two classes have this number of samples) or if a chi-square test, applied on
                the attributes' contingency table has a p-value greater or equal to
                `MAX_P_VALUE_CHI_SQUARE_TEST`. When an attribute is considered invalid for the above
                reasons, this information will be passed to every child node of the current
                TreeNode. Note that numeric attributes are never tested in this way.Defaults to
                `False`.

        Returns:
            A tuple containing, in order:
                - a list of predicted class for each test sample;
                - the number of correct classifications;
                - the number of correct classifications done without test samples with unkown
                    values (that is, values that are unkown at a TreeNode -- they are classified as
                    the most common class at that node);
                - the total cost of the classification errors (when errors costs are uniform, this
                    is equal to the total number of test samples minus the number of correct
                    classifications);
                -  the total cost of the classification errors without considering test
                    samples with unkown values;
                - a list of booleans indicating if the i-th test sample was classified with an
                    unkown value;
                - the number of test samples classified with unkown values;
                - list where the i-th entry has the attribute index used for classification of the
                    i-th sample when an unkown value occurred.
        """

        #TESTED!
        self.train(dataset,
                   training_samples_indices,
                   max_depth,
                   min_samples_per_node,
                   max_p_value,
                   use_stop_conditions)
        return self.test(test_sample_indices)

    def save_tree(self, filepath=None):
        """Saves the tree information: nodes, attributes used to split each one, values to each
        side, etc.

        Arguments:
            filepath (str, optional): file in which to save the tree. If `None`, prints to stdout.
                Defaults to `None`.
        """
        # TESTED!
        # Saves in a txt file or something similar
        def _aux_print_nominal_string(attrib_name, string_values, curr_depth):
            # TESTED!
            ret_string = '|' * curr_depth
            ret_string += '{} in {}:'.format(attrib_name, string_values)
            return ret_string

        def _aux_print_numeric_string(attrib_name, mid_point, inequality, curr_depth):
            # TESTED!
            ret_string = '|' * curr_depth
            ret_string += '{} {} {}:'.format(attrib_name, inequality, mid_point)
            return ret_string

        def _aux_print_split(file_object, tree_node, curr_depth):
            # TESTED!
            if tree_node.is_leaf:
                leaf_class_string = self._dataset.class_int_to_name[
                    tree_node.most_common_int_class]
                string_leaf = '|' * curr_depth + 'CLASS: ' + leaf_class_string
                print(string_leaf, file=file_object)
            else:
                attrib_index = tree_node.node_split.separation_attrib_index
                attrib_name = self._dataset.attrib_names[attrib_index]
                mid_point = tree_node.node_split.mid_point
                if mid_point is not None:
                    # <= mid_point, go left
                    print(_aux_print_numeric_string(attrib_name, mid_point, '<=', curr_depth),
                          file=file_object)
                    _aux_print_split(file_object, tree_node.nodes[0], curr_depth + 1)
                    # > mid_point, go right
                    print(_aux_print_numeric_string(attrib_name, mid_point, '>', curr_depth),
                          file=file_object)
                    _aux_print_split(file_object, tree_node.nodes[1], curr_depth + 1)
                else:
                    for split_values, child_node in zip(tree_node.node_split.splits_values,
                                                        tree_node.nodes):
                        curr_string_values = sorted(
                            [self._dataset.attrib_int_to_value[attrib_index][int_value]
                             for int_value in split_values])
                        print(_aux_print_nominal_string(attrib_name,
                                                        curr_string_values,
                                                        curr_depth),
                              file=file_object)
                        _aux_print_split(file_object, child_node, curr_depth + 1)

        if filepath is None:
            _aux_print_split(sys.stdout, self._root_node, curr_depth=0)
        else:
            with open(filepath, 'w') as tree_output_file:
                _aux_print_split(tree_output_file, self._root_node, curr_depth=0)


class TreeNode(object):
    """Contains information of a certain node of a decision tree.

        It has information about the samples used during training at this node and also about it's
    NodeSplit.

    Args:
        dataset (Dataset): dataset of samples used for training/split generation.
        valid_samples_indices (:obj:'list' of 'int'): indices of samples that should be used for
            training at this node.
        valid_nominal_attribute (:obj:'list' of 'bool'): the i-th entry informs wether the i-th
            attribute is a valid nominal one.
        max_depth_remaining (int): maximum depth that the subtree rooted at this node can have. If
            zero, this node will be a leaf.
        min_samples_per_node (int): minimum number of samples that must be present in order to try
            to create a subtree rooted at this node. If less than this, this node will be a leaf.
        use_stop_conditions (bool, optional): informs wether we should use prunning techniques to
            avoid using attributes with small number of samples (and, thus, avoiding statistical
            anomalies). An attribute will be considered invalid if it contains less than
            `MIN_ALLOWED_IN_TWO_LARGEST` samples in the second largest class (this way at least two
            classes have this number of samples) or if a chi-square test, applied on the attributes'
            contingency table has a p-value greater or equal to `MAX_P_VALUE_CHI_SQUARE_TEST`. When
            an attribute is considered invalid for the above reasons, this information will be
            passed to every child node of the current TreeNode. Note that numeric attributes are
            never tested in this way. Defaults to `False`.
        is_monte_carlo_criterion (bool, optional): indicates if the splitting criterion uses our
            Monte Carlo framework. Defaults to `False`.
        upper_p_value_threshold (float, optional): the p-value-upper-threshold for our Monte Carlo
            framework. If an attribute has a p-value above this threshold, it will be rejected with
            probability `prob_monte_carlo`. Defaults to `None`.
        lower_p_value_threshold (float, optional): the p-value-lower-threshold for our Monte Carlo
            framework. If an attribute has a p-value below this threshold, it will be accepted with
            probability `prob_monte_carlo`. Defaults to `None`.
        prob_monte_carlo (float, optional): the probability of accepting an attribute with p-value
            smaller than `lower_p_value_threshold` and rejecting an attribute with p-value greater
            than `upper_p_value_threshold` for our Monte Carlo framework. Defaults to `None`.
        monte_carlo_t_f_cache (:obj:'dict' of 'tuple' of 'int', optional): cache used to save the
            number of tests and number of fails allowed, given the number of attributes. Defaults to
            `None`.
        calculate_expected_tests (bool, optional): indicates wether we should calculate the expected
            number of tests done by our monte carlo framework. Defaults to `False`.
        monte_carlo_expected_cache (:obj:'dict' of 'tuple' of 'int', optional):  cache used to save
            the number of tests and number of fails allowed, given the number of attributes.
            Defaults to `None`.

    Attributes:
        is_leaf (bool): indicates if the current TreeNode is a tree leaf.
        max_depth_remaining (int): maximum depth that the subtree rooted at the current TreeNode can
            still grow. If zero, the current TreeNode will be a leaf.
        node_split (NodeSplit): Data structure containing information about which attribute and
            split values were obtained in this TreeNode with a certain criterion. Also contains the
            criterion value. It is None if the current TreeNode is a leaf.
        nodes (:obj:'list' of 'TreeNode'): list containing every child TreeNode from the current
            TreeNode.
        contingency_tables (:obj:'list' of 'tuple' of 'list' of 'np.array'): contains a list where
            the i-th entry is a tuple containing two pieces of information of the i-th attribute:
            the contingency table for that attribute (value index is row, class index is column) and
            a list of number of times each value is attained in the training set (i-th entry is the
            number of times a sample has value i in this attribute and training dataset). Used by
            many criteria when calculating the optimal split. Note that, for invalid attributes, the
            entry is a tuple with empty lists ([], []).
        dataset (Dataset): dataset containing the training samples.
        valid_samples_indices (:obj:'list' of 'int'): contains the indices of the valid training
            samples.
        valid_nominal_attribute (:obj:'list' of 'bool'): list where the i-th entry indicates wether
            the i-th attribute from the dataset is valid and nominal or not.
        num_valid_samples (int): number of training samples in this TreeNode.
        class_index_num_samples (:obj:'list' of 'int'): list where the i-th entry indicates the
            number of samples having class i.
        most_common_int_class (int): index of the most frequent class.
        number_non_empty_classes (int): number of classes having no sample in this TreeNode.
        number_samples_in_rarest_class (int): number of samples whose class is the rarest non-empty
            one.
        is_monte_carlo_criterion (bool): indicates if the splitting criterion uses our Monte Carlo
            framework.
        upper_p_value_threshold (float): the p-value-upper-threshold for our Monte Carlo framework.
        lower_p_value_threshold (float): the p-value-lower-threshold for our Monte Carlo framework.
        prob_monte_carlo (float): the probability of accepting an attribute with p-value smaller
            than `lower_p_value_threshold` and rejecting an attribute with p-value greater than
            `upper_p_value_threshold` for our Monte Carlo framework.
        monte_carlo_t_f_cache (:obj:'dict' of 'tuple' of 'int'): cache used to save the number of
            tests and number of fails allowed, given the number of attributes.
        calculate_expected_tests (bool): indicates wether we should calculate the expected number of
            tests done by our monte carlo framework.
        monte_carlo_expected_cache (:obj:'dict' of 'tuple' of 'int'):  cache used to save the number
            of tests and number of fails allowed, given the number of attributes.
        total_expected_num_tests (float): total number of expected tests to be done at this node, in
            the worst-case p-value distribution.
        time_num_tests_fails (float): time taken to calculate the value of
            `total_expected_num_tests`, in seconds.
        time_expected_tests (float): time taken to calculate the value of
            `total_expected_num_tests`, in seconds.
    """
    def __init__(self, dataset, valid_samples_indices, valid_nominal_attribute,
                 max_depth_remaining, min_samples_per_node, use_stop_conditions=False,
                 is_monte_carlo_criterion=False, upper_p_value_threshold=None,
                 lower_p_value_threshold=None, prob_monte_carlo=None, monte_carlo_t_f_cache=None,
                 calculate_expected_tests=False, monte_carlo_expected_cache=None):
        self._use_stop_conditions = use_stop_conditions

        self.is_monte_carlo_criterion = is_monte_carlo_criterion
        self.calculate_expected_tests = calculate_expected_tests
        self.upper_p_value_threshold = upper_p_value_threshold
        self.lower_p_value_threshold = lower_p_value_threshold
        self.prob_monte_carlo = prob_monte_carlo
        if is_monte_carlo_criterion and monte_carlo_t_f_cache is None:
            self.monte_carlo_t_f_cache = {}
        else:
            self.monte_carlo_t_f_cache = monte_carlo_t_f_cache
        if calculate_expected_tests and monte_carlo_expected_cache is None:
            self.monte_carlo_expected_cache = {}
        else:
            self.monte_carlo_expected_cache = monte_carlo_expected_cache
        self.total_expected_num_tests = 0
        self.time_num_tests_fails = 0.0
        self.time_expected_tests = 0.0

        self.max_depth_remaining = max_depth_remaining
        self._min_samples_per_node = min_samples_per_node

        self.is_leaf = True
        self.node_split = None
        self.nodes = []
        self.contingency_tables = None

        self.dataset = dataset
        self.valid_samples_indices = valid_samples_indices
        # Note that self.valid_nominal_attribute might be different from
        # self.dataset.valid_nominal_attribute when use_stop_conditions == True.
        self.valid_nominal_attribute = valid_nominal_attribute

        self.num_valid_samples = len(valid_samples_indices)
        self.class_index_num_samples = [0] * dataset.num_classes
        self.most_common_int_class = None
        self.number_non_empty_classes = 0
        self.number_samples_in_rarest_class = 0 # only among non-empty classes!

        # Fill self.class_index_num_samples
        for sample_index in valid_samples_indices:
            self.class_index_num_samples[
                dataset.sample_class[sample_index]] += 1

        self.most_common_int_class = self.class_index_num_samples.index(
            max(self.class_index_num_samples))

        # Start self.number_samples_in_rarest_class with a > 0 value and we can only decrease it in
        # the loop below to a > 0 value.
        self.number_samples_in_rarest_class = self.class_index_num_samples[
            self.most_common_int_class]
        for class_num_samples in self.class_index_num_samples:
            if class_num_samples > 0:
                self.number_non_empty_classes += 1
                if class_num_samples < self.number_samples_in_rarest_class:
                    self.number_samples_in_rarest_class = class_num_samples

        self._calculate_contingency_tables()


    def _calculate_contingency_tables(self):
        self.contingency_tables = [] # vector of pairs (attrib_contingency_table,
                                     #                  attrib_values_num_samples)
        for (attrib_index,
             is_valid_nominal_attribute) in enumerate(self.valid_nominal_attribute):
            if not is_valid_nominal_attribute:
                self.contingency_tables.append(([], []))
                continue

            attrib_num_values = len(self.dataset.attrib_int_to_value[attrib_index])
            curr_contingency_table = np.zeros((attrib_num_values, self.dataset.num_classes),
                                              dtype=float)
            curr_values_num_samples = np.zeros((attrib_num_values), dtype=float)

            for sample_index in self.valid_samples_indices:
                curr_sample_value = self.dataset.samples[sample_index][attrib_index]
                curr_sample_class = self.dataset.sample_class[sample_index]
                curr_contingency_table[curr_sample_value][curr_sample_class] += 1
                curr_values_num_samples[curr_sample_value] += 1

            self.contingency_tables.append((curr_contingency_table, curr_values_num_samples))

    def _is_attribute_valid(self, attrib_index, min_allowed_in_two_largest,
                            max_p_value_chi_square_test):
        def _get_chi_square_test_p_value(contingency_table, values_num_samples):
            classes_seen = set()
            for value in range(contingency_table.shape[0]):
                for sample_class, num_samples in enumerate(contingency_table[value]):
                    if num_samples > 0 and sample_class not in classes_seen:
                        classes_seen.add(sample_class)
            num_classes = len(classes_seen)
            if num_classes == 1:
                return 0.0

            num_values = sum(num_samples > 0 for num_samples in values_num_samples)
            num_samples = sum(num_samples for num_samples in values_num_samples)
            curr_chi_square_value = 0.0
            for value, value_num_sample in enumerate(values_num_samples):
                if value_num_sample == 0:
                    continue
                for class_index in classes_seen:
                    expected_value_class = (
                        values_num_samples[value] * self.class_index_num_samples[class_index]
                        / num_samples)
                    diff = contingency_table[value][class_index] - expected_value_class
                    curr_chi_square_value += diff * (diff / expected_value_class)
            return 1. - chi2.cdf(x=curr_chi_square_value, df=((num_classes - 1) * (num_values - 1)))


        values_num_samples = self.contingency_tables[attrib_index][1]
        largest = 0
        second_largest = 0
        for num_samples in values_num_samples:
            if num_samples > largest:
                second_largest = largest
                largest = num_samples
            elif num_samples > second_largest:
                second_largest = num_samples
        if second_largest < min_allowed_in_two_largest:
            return False
        chi_square_test_p_value = _get_chi_square_test_p_value(
            self.contingency_tables[attrib_index][0],
            self.contingency_tables[attrib_index][1])
        return chi_square_test_p_value < max_p_value_chi_square_test

    def create_subtree(self, criterion, max_p_value=None):
        '''Given the splitting criterion, creates a tree rooted at the current TreeNode.

        Arguments:
            criterion (Criterion): splitting criterion used to create the tree recursively.
            max_p_value (float, optional): Maximum p-value to be used for max cut methods with
                Janson bounds (values larger than this makes the current node a leaf). Defaults to
                `None`.
        '''

        def _get_values_to_split(splits_values):
            values_to_split = {}
            for split_index, split_values in enumerate(splits_values):
                for value in split_values:
                    values_to_split[value] = split_index
            return values_to_split

        def _get_splits_samples_indices(num_splits, separation_attrib_index, values_to_split,
                                        valid_samples_indices, samples):
            splits_samples_indices = [[] for _ in range(num_splits)]
            for sample_index in valid_samples_indices:
                sample_value_in_split_attrib = samples[sample_index][separation_attrib_index]
                try:
                    splits_samples_indices[values_to_split[
                        sample_value_in_split_attrib]].append(sample_index)
                except KeyError:
                    print('Should not get here. Sample {} has value {} at attribute # {}, '
                          'but this value is unknown to the decision tree.'.format(
                              sample_index,
                              sample_value_in_split_attrib,
                              separation_attrib_index))
                    sys.exit(1)
            return splits_samples_indices

        def _get_numeric_splits_samples_indices(separation_attrib_index, mid_point,
                                                valid_samples_indices, samples):
            splits_samples_indices = [[], []]
            for sample_index in valid_samples_indices:
                sample_value_in_split_attrib = samples[sample_index][separation_attrib_index]
                if sample_value_in_split_attrib <= mid_point:
                    splits_samples_indices[0].append(sample_index)
                else:
                    splits_samples_indices[1].append(sample_index)
            return splits_samples_indices


        # Is it time to stop growing subtrees?
        if (self.max_depth_remaining <= 0
                or self.num_valid_samples < self._min_samples_per_node
                or self.number_non_empty_classes == 1):
            return None

        if self._use_stop_conditions:
            num_valid_attributes = sum(self.dataset.valid_numeric_attribute)
            new_valid_nominal_attribute = self.valid_nominal_attribute[:]
            for (attrib_index,
                 is_valid_nominal_attribute) in enumerate(self.valid_nominal_attribute):
                if is_valid_nominal_attribute:
                    if (self._is_attribute_valid(
                            attrib_index,
                            min_allowed_in_two_largest=MIN_ALLOWED_IN_TWO_LARGEST,
                            max_p_value_chi_square_test=MAX_P_VALUE_CHI_SQUARE_TEST)):
                        num_valid_attributes += 1
                    else:
                        new_valid_nominal_attribute[attrib_index] = False
            self.valid_nominal_attribute = new_valid_nominal_attribute
            if num_valid_attributes == 0:
                return None

        if self.is_monte_carlo_criterion:
            start_time = timeit.default_timer()
            num_valid_nominal_attributes = sum(self.valid_nominal_attribute)
            if num_valid_nominal_attributes in self.monte_carlo_t_f_cache:
                (num_tests, num_fails_allowed) = self.monte_carlo_t_f_cache[
                    num_valid_nominal_attributes]
            else:
                (num_tests, num_fails_allowed) = monte_carlo.get_tests_and_fails_allowed(
                    self.upper_p_value_threshold,
                    self.lower_p_value_threshold,
                    self.prob_monte_carlo,
                    num_valid_nominal_attributes)
                self.monte_carlo_t_f_cache[num_valid_nominal_attributes] = (num_tests,
                                                                            num_fails_allowed)
            self.time_num_tests_fails = timeit.default_timer() - start_time
        else:
            num_tests = 0
            num_fails_allowed = 0

        if self.calculate_expected_tests:
            start_time = timeit.default_timer()
            num_valid_nominal_attributes = sum(self.valid_nominal_attribute)
            if num_valid_nominal_attributes in self.monte_carlo_expected_cache:
                self.total_expected_num_tests = self.monte_carlo_expected_cache[
                    num_valid_nominal_attributes]
            else:
                self.total_expected_num_tests = monte_carlo.get_expected_total_num_tests(
                    num_tests,
                    num_fails_allowed,
                    num_valid_nominal_attributes)
                self.monte_carlo_expected_cache[
                    num_valid_nominal_attributes] = self.total_expected_num_tests
            self.time_expected_tests = timeit.default_timer() - start_time
        else:
            self.total_expected_num_tests = 0.0

        # Get best split
        (separation_attrib_index,
         splits_values,
         criterion_value,
         p_value) = criterion.select_best_attribute_and_split(self, # self is the current TreeNode
                                                              num_tests,
                                                              num_fails_allowed)

        if math.isinf(criterion_value) or (max_p_value is not None and p_value > max_p_value):
            # Stop condition for Max Cut tree: above p_value or no valid attribute index with more
            # than one value (then criterion_value is default, which is +- inf).
            return None

        if self.dataset.valid_numeric_attribute[separation_attrib_index]:
            # NUMERIC ATTRIBUTE
            last_left_value = list(splits_values[0])[0]
            first_right_value = list(splits_values[1])[0]
            mid_point = 0.5 * (last_left_value + first_right_value)
            splits_samples_indices = _get_numeric_splits_samples_indices(
                separation_attrib_index,
                mid_point,
                self.valid_samples_indices,
                self.dataset.samples)
            # Save this node's split information.
            self.node_split = NodeSplit(separation_attrib_index,
                                        None,
                                        None,
                                        criterion,
                                        criterion_value,
                                        p_value,
                                        mid_point)

        else:
            # NOMINAL ATTRIBUTE

            # Calculate a list containing the inverse information of splits_values: here, given a
            # value, we want to know to which split it belongs
            values_to_split = _get_values_to_split(splits_values)

            splits_samples_indices = _get_splits_samples_indices(len(splits_values),
                                                                 separation_attrib_index,
                                                                 values_to_split,
                                                                 self.valid_samples_indices,
                                                                 self.dataset.samples)
            # Save this node's split information.
            self.node_split = NodeSplit(separation_attrib_index,
                                        splits_values,
                                        values_to_split,
                                        criterion,
                                        criterion_value,
                                        p_value)

        # Create subtrees
        self.is_leaf = False
        for curr_split_samples_indices in splits_samples_indices:
            self.nodes.append(TreeNode(self.dataset,
                                       curr_split_samples_indices,
                                       self.valid_nominal_attribute,
                                       self.max_depth_remaining - 1,
                                       self._min_samples_per_node,
                                       self._use_stop_conditions,
                                       self.is_monte_carlo_criterion,
                                       self.upper_p_value_threshold,
                                       self.lower_p_value_threshold,
                                       self.prob_monte_carlo,
                                       self.monte_carlo_t_f_cache,
                                       self.calculate_expected_tests,
                                       self.monte_carlo_expected_cache))

            self.nodes[-1].create_subtree(criterion, max_p_value)

    def get_most_popular_subtree(self):
        '''Returns the number of samples in the most popular subtree. It should NOT be called in a
        leaf node (otherwise the program will exit).'''
        return max(subtree.num_valid_samples for subtree in self.nodes)

    def get_subtree_time_num_tests_fails(self):
        '''Returns the total time taken to calculate the number of tests and fails allowed at each
        node in this subtree (including the current TreeNode).'''
        if self.is_leaf:
            return self.time_num_tests_fails
        return sum(subtree.get_subtree_time_num_tests_fails() for subtree in self.nodes)

    def get_subtree_time_expected_tests(self):
        '''Returns the total time taken to calculate the total expected number of tests at each node
        in this subtree (including the current TreeNode).'''
        if self.is_leaf:
            return self.time_expected_tests
        return sum(subtree.get_subtree_time_expected_tests() for subtree in self.nodes)

    def is_trivial(self):
        '''Returns a bool indicating if every leaf in the tree starting at TreeNode has the same
        classification. In other words, informs wether we could remove this whole tree and make the
        current TreeNode a leaf, without changing the classifications' outcomes.'''
        def _has_different_class(node, first_class_seen):
            if node.is_leaf:
                return node.most_common_int_class != first_class_seen
            for child_subtree in node.nodes:
                if _has_different_class(child_subtree, first_class_seen):
                    return True
            return False

        if self.is_leaf:
            return True
        first_class_seen = None
        curr_node = self
        while not curr_node.is_leaf:
            curr_node = curr_node.nodes[0]
        first_class_seen = curr_node.most_common_int_class
        return not _has_different_class(self, first_class_seen)

    def get_nodes_infos(self, max_depth=3):
        '''Get information about nodes in the tree, up to `max_depth`.

        Used in some experiments to see which splits are obtained using different criteria.

        Arguments:
            max_depth (int, optional): Maximum depth to which we want information about every node.
                Defaults to `3`.

        Returns:
            A list containing, in order:
                - A list containing the NodeSplit's for every node starting at the current TreeNode
                    all the way to `max_depth` depth;
                - the number of nodes in the tree rooted at the current NodeSplit.
        '''
        return [self.get_nodes_attributes(max_depth), self.get_num_nodes()]

    def get_nodes_attributes(self, max_depth_remaining=3):
        ''' Get information about nodes in the tree, up to `max_depth`.

        Used in some experiments to see which splits are obtained using different criteria.

        Arguments:
            max_depth (int, optional): Maximum depth to which we want information about every node.
                Defaults to `3`.

        Returns:
            A list containing the NodeSplit's for every node starting at the current TreeNode all
            the way to `max_depth` depth.
        '''
        def _get_info_aux(node, max_depth_remaining, ret):
            if max_depth_remaining == 0:
                return
            if node.node_split is None or node.node_split.separation_attrib_index is None:
                ret += [(None, None, True)] * (2**max_depth_remaining - 1)
                return
            max_split_ratio = node.get_most_popular_subtree() / node.num_valid_samples
            ret.append((node.node_split.separation_attrib_index,
                        max_split_ratio,
                        self.is_trivial()))
            _get_info_aux(node.nodes[0], max_depth_remaining - 1, ret)
            _get_info_aux(node.nodes[1], max_depth_remaining - 1, ret)

        ret = []
        _get_info_aux(self, max_depth_remaining, ret)
        return ret

    def get_num_nodes(self):
        '''Returns the number of nodes in the tree rooted at the current TreeNode (counting includes
        leaves and the current TreeNode).'''
        num_nodes = 1
        for child_node in self.nodes:
            num_nodes += child_node.get_num_nodes()
        return num_nodes

    def get_max_depth(self):
        '''Returns the maximum depth of the tree rooted at the current TreeNode. If the current node
        is a leaf, it will return zero.'''
        if self.is_leaf:
            return 0
        ret = 0
        for child_node in self.nodes:
            max_child_depth = child_node.get_max_depth()
            if max_child_depth > ret:
                ret = max_child_depth
        return ret + 1


class NodeSplit(object):
    """Data structure containing information about the best split found on a node.

    Used for debugging and for the classification of test samples.

    Args:
        separation_attrib_index (int): Index of the attribute used for splitting.
        splits_values (:obj:'list' of 'set' of 'int'): list containing a set of attribute values for
            each TreeNode child. Binary splits have two sets (left and right split values), multiway
            splits may have many more.
        values_to_split (:obj:'dict' of 'int'): reversed index for `splits_values`. Given a value,
            it returns the index of the split that this value belongs to.
        criterion (Criterion): criterion used to generate the current split.
        criterion_value (float): optimal criterion value obtained for this TreeNode.
        p_value (float): split p-value according to Janson theorem. Used only for some max cut
            criteria. Defaults to `None`.
        mid_point (float): cut point for numeric splits. Will be the average between the largest
            value on the left split and the smallest value on the right split. Not used for splits
            that use nominal attributes. Defaults to `None`.


    Attributes:
        separation_attrib_index (int): Index of the attribute used for splitting.
        splits_values (:obj:'list' of 'set' of 'int'): list containing a set of attribute values for
            each TreeNode child. Binary splits have two sets (left and right split values), multiway
            splits may have many more.
        values_to_split (:obj:'dict' of 'int'): reversed index for `splits_values`. Given a value,
            it returns the index of the split that this value belongs to.
        mid_point (float): cut point for numeric splits. Will be the average between the largest
            value on the left split and the smallest value on the right split.
        gini_value (float): criterion value for this split. It is used by every criterion that is
            not a max cut one, and not only by gini.
        cut_gain (float): criterion value for this split. It is only used by max cut criteria.
        p_value (float): split p-value according to Janson theorem. Used only for some max cut
            criteria.
    """
    def __init__(self, separation_attrib_index, splits_values, values_to_split, criterion,
                 criterion_value, p_value=None, mid_point=None):
        #TESTED!
        self.separation_attrib_index = separation_attrib_index
        self.splits_values = splits_values
        self.values_to_split = values_to_split
        self.mid_point = mid_point
        if criterion.name == 'Gini Index':
            self.gini_value = criterion_value
        elif criterion.name == 'Gini Twoing':
            self.gini_value = criterion_value
        elif criterion.name == 'Gini Twoing Monte Carlo':
            self.gini_value = criterion_value
        elif criterion.name == 'Twoing':
            self.gini_value = criterion_value
        elif criterion.name == 'ORT':
            self.gini_value = criterion_value
        elif criterion.name == 'MPI':
            self.gini_value = criterion_value
        elif criterion.name == 'Max Cut Exact':
            self.gini_value = criterion_value
        elif criterion.name == 'Max Cut Exact Chi Square':
            self.gini_value = criterion_value
        elif criterion.name == 'Max Cut Exact Chi Square Heuristic':
            self.gini_value = criterion_value
        elif criterion.name == 'Max Cut Exact Residue':
            self.gini_value = criterion_value
        elif criterion.name == 'Gain Ratio':
            self.sep_gain = criterion_value
        elif criterion.name == 'Max Cut':
            self.cut_gain = criterion_value
            self.p_value = p_value
        elif criterion.name == 'Max Cut Naive':
            self.cut_gain = criterion_value
        elif criterion.name == 'Max Cut Naive With Local Search':
            self.cut_gain = criterion_value
        elif criterion.name == 'Fast Max Cut Naive':
            self.cut_gain = criterion_value
        elif criterion.name == 'Max Cut Naive Residue':
            self.cut_gain = criterion_value
        elif criterion.name == 'Max Cut Naive Chi Square':
            self.cut_gain = criterion_value
        elif criterion.name == 'Max Cut Naive Chi Square With Local Search':
            self.cut_gain = criterion_value
        elif criterion.name == 'Fast Max Cut Chi Square':
            self.cut_gain = criterion_value
        elif criterion.name == 'Max Cut Naive Chi Square Normalized':
            self.cut_gain = criterion_value
        elif criterion.name == 'Max Cut Naive Chi Square Normalized With Local Search':
            self.cut_gain = criterion_value
        elif criterion.name == 'Fast Max Cut Chi Square Normalized':
            self.cut_gain = criterion_value
        elif criterion.name == 'Fast Max Cut Chi Square Normalized P Value':
            self.cut_gain = criterion_value
        elif criterion.name == 'Fast Max Cut Chi Square Normalized P Value M C':
            self.cut_gain = criterion_value
        elif criterion.name == 'Max Cut Naive Chi Square Heuristic':
            self.cut_gain = criterion_value
        elif criterion.name == 'Max Cut Monte Carlo':
            self.cut_gain = criterion_value
        elif criterion.name == 'Max Cut Monte Carlo Residue':
            self.cut_gain = criterion_value
        else:
            print('NodeSplit with unknown criterion name: {}'.format(criterion.name))
            sys.exit(1)
