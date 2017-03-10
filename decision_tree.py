#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import random
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import chi2


MIN_ALLOWED_IN_TWO_LARGEST = 40
MAX_P_VALUE_CHI_SQUARE_TEST = 0.1


class DecisionTree(object):
    def __init__(self, criterion):
        #TESTED!
        self._criterion = criterion
        self._dataset = None
        self._root_node = None

    def get_root_node(self):
        return self._root_node

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
        #TESTED!
        self._dataset = dataset
        print('Starting tree training...')
        self._root_node = TreeNode(dataset,
                                   training_samples_indices,
                                   dataset.valid_nominal_attribute[:],
                                   max_depth,
                                   min_samples_per_node,
                                   use_stop_conditions)
        self._root_node.create_subtree(self._criterion, max_p_value)
        print('Done!')

    def train_and_self_validate(self, dataset, training_samples_indices,
                                validation_sample_indices, max_depth, min_samples_per_node,
                                max_p_value=None, use_stop_conditions=False):
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
        #TESTED!
        self.train(dataset,
                   training_samples_indices,
                   max_depth,
                   min_samples_per_node,
                   max_p_value,
                   use_stop_conditions)
        return self.test(test_sample_indices)

    def save_tree(self, filepath=None):
        # TESTED!
        # Saves in a txt file or something similar
        def aux_print_nominal_string(attrib_name, string_values, curr_depth):
            # TESTED!
            ret_string = '|' * curr_depth
            ret_string += '{} in {}:'.format(attrib_name, string_values)
            return ret_string

        def aux_print_numeric_string(attrib_name, mid_point, inequality, curr_depth):
            # TESTED!
            ret_string = '|' * curr_depth
            ret_string += '{} {} {}:'.format(attrib_name, inequality, mid_point)
            return ret_string

        def aux_print_split(file_object, tree_node, curr_depth):
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
                    print(aux_print_numeric_string(attrib_name, mid_point, '<=', curr_depth),
                          file=file_object)
                    aux_print_split(file_object, tree_node.nodes[0], curr_depth + 1)
                    # > mid_point, go right
                    print(aux_print_numeric_string(attrib_name, mid_point, '>', curr_depth),
                          file=file_object)
                    aux_print_split(file_object, tree_node.nodes[1], curr_depth + 1)
                else:
                    for split_values, child_node in zip(tree_node.node_split.splits_values,
                                                        tree_node.nodes):
                        curr_string_values = sorted(
                            [self._dataset.attrib_int_to_value[attrib_index][int_value]
                             for int_value in split_values])
                        print(aux_print_nominal_string(attrib_name, curr_string_values, curr_depth),
                              file=file_object)
                        aux_print_split(file_object, child_node, curr_depth + 1)

        if filepath is None:
            aux_print_split(sys.stdout, self._root_node, curr_depth=0)
        else:
            with open(filepath, 'w') as tree_output_file:
                aux_print_split(tree_output_file, self._root_node, curr_depth=0)


class TreeNode(object):
    def __init__(self, dataset, valid_samples_indices, valid_nominal_attribute,
                 max_depth_remaining, min_samples_per_node, use_stop_conditions=False):
        self.use_stop_conditions = use_stop_conditions
        self.is_leaf = True
        self.max_depth_remaining = max_depth_remaining
        self.min_samples_per_node = min_samples_per_node
        self.node_split = None
        self.nodes = []
        self.contingency_tables = None

        self.dataset = dataset
        self.valid_samples_indices = valid_samples_indices
        self.valid_nominal_attribute = valid_nominal_attribute

        self.num_valid_samples = len(valid_samples_indices)
        self.most_common_int_class = None
        self.class_index_num_samples = [0] * dataset.num_classes
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

        self.calculate_contingency_tables()


    def calculate_contingency_tables(self):
        self.contingency_tables = [] # vector of pairs (attrib_contingency_table,
                                     #                  attrib_values_num_samples)
        for (attrib_index,
             is_valid_nominal_attribute) in enumerate(self.dataset.valid_nominal_attribute):
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

    def is_attribute_valid(self, attrib_index, min_allowed_in_two_largest,
                           max_p_value_chi_square_test):
        def get_chi_square_test_p_value(contingency_table, values_num_samples):
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
        chi_square_test_p_value = get_chi_square_test_p_value(
            self.contingency_tables[attrib_index][0],
            self.contingency_tables[attrib_index][1])
        return chi_square_test_p_value < max_p_value_chi_square_test

    def create_subtree(self, criterion, max_p_value=None):

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
                except KeyError as e:
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
                or self.num_valid_samples < self.min_samples_per_node
                or self.number_non_empty_classes == 1):
            return None

        if self.use_stop_conditions:
            num_valid_attributes = sum(self.dataset.valid_numeric_attribute)
            new_valid_nominal_attribute = self.valid_nominal_attribute[:]
            for (attrib_index,
                 is_valid_nominal_attribute) in enumerate(self.valid_nominal_attribute):
                if is_valid_nominal_attribute:
                    if (self.is_attribute_valid(
                            attrib_index,
                            min_allowed_in_two_largest=MIN_ALLOWED_IN_TWO_LARGEST,
                            max_p_value_chi_square_test=MAX_P_VALUE_CHI_SQUARE_TEST)):
                        num_valid_attributes += 1
                    else:
                        new_valid_nominal_attribute[attrib_index] = False
            self.valid_nominal_attribute = new_valid_nominal_attribute
            if num_valid_attributes == 0:
                return None

        # Get best split
        (separation_attrib_index,
         splits_values,
         criterion_value,
         p_value) = criterion.select_best_attribute_and_split(self) # self is the current TreeNode

        if math.isinf(criterion_value) or (max_p_value is not None and p_value > max_p_value):
            # Stop condition for Max Cut tree: above p_value or no valid attribute index with more
            # than one value (then criterion_value is default, which is +- inf).
            return None

        if self.dataset.valid_numeric_attribute[separation_attrib_index]:
            # NUMERIC ATTRIBUTE
            last_left_value = list(splits_values[0])[0]
            first_right_value = list(splits_values[1])[0]
            mid_point = 0.5 * (last_left_value + first_right_value)
            splits_samples_indices = _get_numeric_splits_samples_indices(separation_attrib_index,
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
                                       self.min_samples_per_node,
                                       self.use_stop_conditions))

            self.nodes[-1].create_subtree(criterion, max_p_value)

    def get_most_popular_subtree(self):
        return max(subtree.num_valid_samples for subtree in self.nodes)

    def is_trivial(self):
        def has_different_class(node, first_class_seen):
            if node.is_leaf:
                return node.most_common_int_class != first_class_seen
            for child_subtree in node.nodes:
                if has_different_class(child_subtree, first_class_seen):
                    return True
            return False

        if self.is_leaf:
            return True
        first_class_seen = None
        curr_node = self
        while not curr_node.is_leaf:
            curr_node = curr_node.nodes[0]
        first_class_seen = curr_node.most_common_int_class
        return not has_different_class(self, first_class_seen)

    def get_nodes_infos(self, max_depth=3):
        return [self.get_nodes_attributes(max_depth), self.get_num_nodes()]

    def get_nodes_attributes(self, max_depth_remaining=3):
        def get_info_aux(node, max_depth_remaining, ret):
            if max_depth_remaining == 0:
                return
            if node.node_split is None or node.node_split.separation_attrib_index is None:
                ret += [(None, None, True)] * (2**max_depth_remaining - 1)
                return
            max_split_ratio = node.get_most_popular_subtree() / node.num_valid_samples
            ret.append((node.node_split.separation_attrib_index,
                        max_split_ratio,
                        self.is_trivial()))
            get_info_aux(node.nodes[0], max_depth_remaining - 1, ret)
            get_info_aux(node.nodes[1], max_depth_remaining - 1, ret)

        ret = []
        get_info_aux(self, max_depth_remaining, ret)
        return ret

    def get_num_nodes(self):
        num_nodes = 1
        for child_node in self.nodes:
            num_nodes += child_node.get_num_nodes()
        return num_nodes

    def get_max_depth(self):
        if self.is_leaf:
            return 0
        ret = 0
        for child_node in self.nodes:
            max_child_depth = child_node.get_max_depth()
            if max_child_depth > ret:
                ret = max_child_depth
        return ret + 1


class NodeSplit(object):
    def __init__(self, separation_attrib_index, splits_values, values_to_split, criterion,
                 criterion_value, p_value, mid_point=None):
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
