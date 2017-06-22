#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing all criteria available for tests.


"""

import abc
import collections
import itertools
import math
import random

import numpy as np
import scipy


#: Whether Monte Carlo Framework should order attributes randomly or in decreasing criterion value.
ORDER_RANDOMLY = False

#: Whether Monte Carlo Framework should only test the best attribute for each number of values. Only
#: works when `ORDER_RANDOMLY` is `True`.
USE_ONE_ATTRIB_PER_NUM_VALUES = False

#: Holds the last-calculated list of the p-values of each attribute, calculated only when using
#: Monte Carlo criteria.
LAST_P_VALUES = None # contains (curr_position, best_attrib_split, curr_p_value)

#: Number of monte carlo simulations to run with Monte Carlo criteria.
NUM_MONTE_CARLO_SIMULATIONS = 100

#: Contains the information about a given split. When empty, defaults to
#: `(None, [], float('-inf'))`.
Split = collections.namedtuple('Split',
                               ['attrib_index',
                                'splits_values',
                                'criterion_value'])
Split.__new__.__defaults__ = (None, [], float('-inf'))


class Criterion(object):
    """Abstract base class for every criterion.
    """
    __metaclass__ = abc.ABCMeta

    name = ''

    @classmethod
    @abc.abstractmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best split found, according to the criterion, using `num_tests` tests per
        attribute and accepting if it doesn't fail more than `num_fails_allowed` times.
        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.
        """
        # returns (best_split, num_tests_needed, position_of_accepted)
        pass



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                        GINI GAIN                                          ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GiniGain(Criterion):
    """Gini Gain criterion. For reference see "Breiman, L., Friedman, J. J., Olshen, R. A., and
    Stone, C. J. Classification and Regression Trees. Wadsworth, 1984".
    """
    name = 'Gini Gain'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the Gini Gain criterion,
        using `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times. If `num_tests` is zero, returns the attribute/split with
        the largest criterion value. Note that this criteria gives a multiway-split.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.

        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """
        original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                  tree_node.class_index_num_samples)
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_children_gini_index = cls._calculate_children_gini_index(
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    len(tree_node.valid_samples_indices),)
                curr_total_gini_gain = original_gini - curr_children_gini_index
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_total_gini_gain))
        if num_tests == 0: # Just return attribute/split with maximum Gini Gain.
            if best_splits_per_attrib:
                best_split = max(best_splits_per_attrib, key=lambda split: split.criterion_value)
            else:
                best_split = Split()
            num_monte_carlo_tests_needed = 0
            position_of_accepted = 1
            return (best_split, num_monte_carlo_tests_needed, position_of_accepted)
        else: # use Monte Carlo approach.
            if ORDER_RANDOMLY:
                random.shuffle(best_splits_per_attrib)
            else:
                best_splits_per_attrib.sort(key=lambda split: -split.criterion_value)
                if USE_ONE_ATTRIB_PER_NUM_VALUES:
                    best_splits_per_attrib_clean = []
                    num_values_seen = set()
                    for curr_split in best_splits_per_attrib:
                        num_values = len(curr_split.splits_values)
                        if num_values not in num_values_seen:
                            num_values_seen.add(num_values)
                            best_splits_per_attrib_clean.append(curr_split)
                    best_splits_per_attrib = best_splits_per_attrib_clean

            total_num_tests_needed = 0
            for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
                (should_accept,
                 num_tests_needed) = cls._accept_attribute(
                     best_attrib_split.criterion_value,
                     num_tests,
                     num_fails_allowed,
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.contingency_tables[
                         best_attrib_split.attrib_index].values_num_samples)
                total_num_tests_needed += num_tests_needed
                if should_accept:
                    return (best_attrib_split, total_num_tests_needed, curr_position + 1)
            return (Split(), total_num_tests_needed, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_children_gini_index(cls, contingency_table, values_num_seen, num_valid_samples):
        total_children_gini = 0.0
        for value_index, value_num_samples in enumerate(values_num_seen):
            if value_num_samples == 0:
                continue
            curr_child_gini_index = cls._calculate_gini_index(value_num_samples,
                                                              contingency_table[value_index, :])
            total_children_gini += (value_num_samples / num_valid_samples) * curr_child_gini_index
        return total_children_gini

    @staticmethod
    def _calculate_gini_index(num_samples, class_num_samples):
        gini_index = 1.0
        for curr_class_num_samples in class_num_samples:
            if curr_class_num_samples > 0:
                gini_index -= (curr_class_num_samples / num_samples)**2
        return gini_index

    @staticmethod
    def _generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
        # TESTED!
        random_classes = np.random.choice(len(classes_dist),
                                          num_valid_samples,
                                          replace=True,
                                          p=classes_dist)
        random_contingency_table = np.zeros((values_num_samples.shape[0], len(classes_dist)),
                                            dtype=float)
        samples_done = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples > 0:
                for class_index in random_classes[samples_done: samples_done + value_num_samples]:
                    random_contingency_table[value, class_index] += 1
                samples_done += value_num_samples
        return random_contingency_table

    @classmethod
    def _accept_attribute(cls, real_gini_gain, num_tests, num_fails_allowed, num_valid_samples,
                          class_index_num_samples, values_num_samples):
        num_classes = len(class_index_num_samples)
        classes_dist = class_index_num_samples[:]
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)

        num_fails_seen = 0
        for test_number in range(1, num_tests + 1):
            random_contingency_table = cls._generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                values_num_samples)

            new_class_index_num_samples = np.sum(random_contingency_table, axis=0).tolist()
            father_gini_index = cls._calculate_gini_index(num_valid_samples,
                                                          new_class_index_num_samples)
            curr_children_gini_index = cls._calculate_children_gini_index(
                random_contingency_table,
                values_num_samples,
                num_valid_samples)
            curr_total_gini_gain = father_gini_index - curr_children_gini_index

            if curr_total_gini_gain > real_gini_gain:
                num_fails_seen += 1
                if num_fails_seen > num_fails_allowed:
                    return False, test_number
            if num_tests - test_number <= num_fails_allowed - num_fails_seen:
                return True, test_number
        return True, num_tests



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       TWOING                                              ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class Twoing(Criterion):
    """Twoing criterion. For reference see "Breiman, L., Friedman, J. J., Olshen, R. A., and
    Stone, C. J. Classification and Regression Trees. Wadsworth, 1984".
    """
    name = 'Twoing'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the Twoing criterion, using
        `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times. If `num_tests` is zero, returns the attribute/split with the
        largest criterion value.
        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.
        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """
        best_splits_per_attrib = []
        values_seen_per_attrib = []
        for attrib_index, is_valid_nominal_attrib in enumerate(tree_node.valid_nominal_attribute):
            if not is_valid_nominal_attrib:
                values_seen_per_attrib.append(None)
                continue
            else:
                best_total_gini_gain = float('-inf')
                best_left_values = set()
                best_right_values = set()
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                values_seen_per_attrib.append(values_seen)
                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index].contingency_table,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         set_left_classes,
                         set_right_classes)
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    (curr_gini_gain,
                     left_values,
                     right_values) = cls._two_class_trick(
                         original_gini,
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    if curr_gini_gain > best_total_gini_gain:
                        best_total_gini_gain = curr_gini_gain
                        best_left_values = left_values
                        best_right_values = right_values
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=[best_left_values,
                                                                   best_right_values],
                                                    criterion_value=best_total_gini_gain))


        if num_tests == 0: # Just return attribute/split with maximum Gini Gain.
            if best_splits_per_attrib:
                best_split = max(best_splits_per_attrib, key=lambda split: split.criterion_value)
            else:
                best_split = Split()
            num_monte_carlo_tests_needed = 0
            position_of_accepted = 1
            return (best_split, num_monte_carlo_tests_needed, position_of_accepted)
        else: # use Monte Carlo approach.
            if ORDER_RANDOMLY:
                random.shuffle(best_splits_per_attrib)
            else:
                best_splits_per_attrib.sort(key=lambda split: -split.criterion_value)
                if USE_ONE_ATTRIB_PER_NUM_VALUES:
                    best_splits_per_attrib_clean = []
                    num_values_seen = set()
                    for curr_split in best_splits_per_attrib:
                        num_values = len(curr_split.splits_values)
                        if num_values not in num_values_seen:
                            num_values_seen.add(num_values)
                            best_splits_per_attrib_clean.append(curr_split)
                    best_splits_per_attrib = best_splits_per_attrib_clean

            total_num_tests_needed = 0
            for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
                (should_accept,
                 num_tests_needed) = cls._accept_attribute(
                     best_attrib_split.criterion_value,
                     num_tests,
                     num_fails_allowed,
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.contingency_tables[
                         best_attrib_split.attrib_index].values_num_samples,
                     values_seen_per_attrib[best_attrib_split.attrib_index])
                total_num_tests_needed += num_tests_needed
                if should_accept:
                    return (best_attrib_split, total_num_tests_needed, curr_position + 1)
            return (Split(), total_num_tests_needed, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_twoing(class_index_num_samples):
        # We only need to look at superclasses of up to (len(class_index_num_samples)/2 + 1)
        # elements because of symmetry! The subsets we are not choosing are complements of the ones
        # chosen.
        non_empty_classes = set([])
        for class_index, class_num_samples in enumerate(class_index_num_samples):
            if class_num_samples > 0:
                non_empty_classes.add(class_index)
        number_non_empty_classes = len(non_empty_classes)

        for left_classes in itertools.chain.from_iterable(
                itertools.combinations(non_empty_classes, size_left_superclass)
                for size_left_superclass in range(1, number_non_empty_classes // 2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if not set_left_classes or not set_right_classes:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_twoing_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                      set_right_classes):
        twoing_contingency_table = np.zeros((contingency_table.shape[0], 2), dtype=float)
        superclass_index_num_samples = [0, 0]
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index in set_left_classes:
                superclass_index_num_samples[0] += contingency_table[value][class_index]
                twoing_contingency_table[value][0] += contingency_table[value][class_index]
            for class_index in set_right_classes:
                superclass_index_num_samples[1] += contingency_table[value][class_index]
                twoing_contingency_table[value][1] += contingency_table[value][class_index]
        return twoing_contingency_table, superclass_index_num_samples

    @staticmethod
    def _two_class_trick(original_gini, class_index_num_samples, values_seen, values_num_samples,
                         contingency_table, num_total_valid_samples):
        # TESTED!
        def _get_non_empty_class_indices(class_index_num_samples):
            # TESTED!
            first_non_empty_class = None
            second_non_empty_class = None
            for class_index, class_num_samples in enumerate(class_index_num_samples):
                if class_num_samples > 0:
                    if first_non_empty_class is None:
                        first_non_empty_class = class_index
                    else:
                        second_non_empty_class = class_index
                        break
            return first_non_empty_class, second_non_empty_class

        def _calculate_value_class_ratio(values_seen, values_num_samples, contingency_table,
                                         non_empty_class_indices):
            # TESTED!
            value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = contingency_table[curr_value][second_class_index]
                value_number_ratio.append((curr_value,
                                           number_second_non_empty,
                                           number_second_non_empty/values_num_samples[curr_value]))
            value_number_ratio.sort(key=lambda tup: tup[2])
            return value_number_ratio

        def _calculate_children_gini_index(num_left_first, num_left_second, num_right_first,
                                           num_right_second, num_left_samples, num_right_samples):
            # TESTED!
            if num_left_samples != 0:
                left_first_class_freq_ratio = float(num_left_first)/float(num_left_samples)
                left_second_class_freq_ratio = float(num_left_second)/float(num_left_samples)
                left_split_gini_index = (1.0
                                         - left_first_class_freq_ratio**2
                                         - left_second_class_freq_ratio**2)
            else:
                # We can set left_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                left_split_gini_index = 1.0

            if num_right_samples != 0:
                right_first_class_freq_ratio = float(num_right_first)/float(num_right_samples)
                right_second_class_freq_ratio = float(num_right_second)/float(num_right_samples)
                right_split_gini_index = (1.0
                                          - right_first_class_freq_ratio**2
                                          - right_second_class_freq_ratio**2)
            else:
                # We can set right_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                right_split_gini_index = 1.0

            curr_children_gini_index = ((num_left_samples * left_split_gini_index
                                         + num_right_samples * right_split_gini_index)
                                        / (num_left_samples + num_right_samples))
            return curr_children_gini_index

        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_class,
         second_non_empty_class) = _get_non_empty_class_indices(class_index_num_samples)
        if first_non_empty_class is None or second_non_empty_class is None:
            return (float('-inf'), {0}, set())

        value_number_ratio = _calculate_value_class_ratio(values_seen,
                                                          values_num_samples,
                                                          contingency_table,
                                                          (first_non_empty_class,
                                                           second_non_empty_class))

        best_split_total_gini_gain = float('-inf')
        best_last_left_index = 0

        num_left_first = 0
        num_left_second = 0
        num_left_samples = 0
        num_right_first = class_index_num_samples[first_non_empty_class]
        num_right_second = class_index_num_samples[second_non_empty_class]
        num_right_samples = num_total_valid_samples

        for last_left_index, (last_left_value, last_left_num_second, _) in enumerate(
                value_number_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            last_left_num_first = num_samples_last_left_value - last_left_num_second

            num_left_samples += num_samples_last_left_value
            num_left_first += last_left_num_first
            num_left_second += last_left_num_second
            num_right_samples -= num_samples_last_left_value
            num_right_first -= last_left_num_first
            num_right_second -= last_left_num_second

            curr_children_gini_index = _calculate_children_gini_index(num_left_first,
                                                                      num_left_second,
                                                                      num_right_first,
                                                                      num_right_second,
                                                                      num_left_samples,
                                                                      num_right_samples)
            curr_gini_gain = original_gini - curr_children_gini_index
            if curr_gini_gain > best_split_total_gini_gain:
                best_split_total_gini_gain = curr_gini_gain
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_gain, set_left_values, set_right_values)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side/side_num)**2
        return gini_index

    @classmethod
    def _calculate_children_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        children_gini_index = ((left_num * left_split_gini_index
                                + right_num * right_split_gini_index)
                               / (left_num + right_num))
        return children_gini_index

    @staticmethod
    def _generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
        # TESTED!
        random_classes = np.random.choice(len(classes_dist),
                                          num_valid_samples,
                                          replace=True,
                                          p=classes_dist)
        random_contingency_table = np.zeros((values_num_samples.shape[0], len(classes_dist)),
                                            dtype=float)
        samples_done = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples > 0:
                for class_index in random_classes[samples_done: samples_done + value_num_samples]:
                    random_contingency_table[value, class_index] += 1
                samples_done += value_num_samples
        return random_contingency_table

    @classmethod
    def _accept_attribute(cls, real_gini_gain, num_tests, num_fails_allowed, num_valid_samples,
                          class_index_num_samples, values_num_samples, values_seen):
        num_classes = len(class_index_num_samples)
        classes_dist = class_index_num_samples[:]
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)

        num_fails_seen = 0
        for test_number in range(1, num_tests + 1):
            random_contingency_table = cls._generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                values_num_samples)

            best_gini_gain = float('-inf')
            for (set_left_classes,
                 set_right_classes) in cls._generate_twoing(class_index_num_samples):

                (twoing_contingency_table,
                 superclass_index_num_samples) = cls._get_twoing_contingency_table(
                     random_contingency_table,
                     values_num_samples,
                     set_left_classes,
                     set_right_classes)
                original_gini = cls._calculate_gini_index(num_valid_samples,
                                                          superclass_index_num_samples)
                (curr_gini_gain, _, _) = cls._two_class_trick(original_gini,
                                                              superclass_index_num_samples,
                                                              values_seen,
                                                              values_num_samples,
                                                              twoing_contingency_table,
                                                              num_valid_samples)
                if curr_gini_gain > best_gini_gain:
                    best_gini_gain = curr_gini_gain

            if best_gini_gain > real_gini_gain:
                num_fails_seen += 1
                if num_fails_seen > num_fails_allowed:
                    return False, test_number
            if num_tests - test_number <= num_fails_allowed - num_fails_seen:
                return True, test_number
        return True, num_tests



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       GAIN RATIO                                          ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GainRatio(Criterion):
    """Gain Ratio criterion. For reference see "Quinlan, J. R. C4.5: Programs for Machine Learning.
    Morgan Kaufmann Publishers, 1993.".
    """
    name = 'Gain Ratio'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the Gain Ratio criterion,
        using `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times. If `num_tests` is zero, returns the attribute/split with the
        largest criterion value.
        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.
        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """

        # First we calculate the original class frequency and information
        original_information = cls._calculate_information(tree_node.class_index_num_samples,
                                                          len(tree_node.valid_samples_indices))
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_gain_ratio = cls._calculate_gain_ratio(
                    len(tree_node.valid_samples_indices),
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    original_information)
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_gain_ratio))

        if num_tests == 0: # Just return attribute/split with maximum Gain Ratio.
            if best_splits_per_attrib:
                best_split = max(best_splits_per_attrib, key=lambda split: split.criterion_value)
            else:
                best_split = Split()
            num_monte_carlo_tests_needed = 0
            position_of_accepted = 1
            return (best_split, num_monte_carlo_tests_needed, position_of_accepted)
        else: # use Monte Carlo approach.
            if ORDER_RANDOMLY:
                random.shuffle(best_splits_per_attrib)
            else:
                best_splits_per_attrib.sort(key=lambda split: -split.criterion_value)
                if USE_ONE_ATTRIB_PER_NUM_VALUES:
                    best_splits_per_attrib_clean = []
                    num_values_seen = set()
                    for curr_split in best_splits_per_attrib:
                        num_values = len(curr_split.splits_values)
                        if num_values not in num_values_seen:
                            num_values_seen.add(num_values)
                            best_splits_per_attrib_clean.append(curr_split)
                    best_splits_per_attrib = best_splits_per_attrib_clean

            total_num_tests_needed = 0
            for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
                (should_accept,
                 num_tests_needed) = cls._accept_attribute(
                     best_attrib_split.criterion_value,
                     num_tests,
                     num_fails_allowed,
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.contingency_tables[
                         best_attrib_split.attrib_index].values_num_samples)
                total_num_tests_needed += num_tests_needed
                if should_accept:
                    return (best_attrib_split, total_num_tests_needed, curr_position + 1)
            return (Split(), total_num_tests_needed, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_gain_ratio(cls, num_valid_samples, contingency_table, values_num_samples,
                              original_information):
        information_gain = original_information # Initial Information Gain
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            curr_split_information = cls._calculate_information(contingency_table[value],
                                                                value_num_samples)
            information_gain -= (value_num_samples / num_valid_samples) * curr_split_information

        # Gain Ratio
        potential_partition_information = cls._calculate_potential_information(values_num_samples,
                                                                               num_valid_samples)
        # Note that, since there are at least two different values, potential_partition_information
        # is never zero.
        gain_ratio = information_gain / potential_partition_information
        return gain_ratio

    @staticmethod
    def _calculate_information(class_index_num_samples, num_valid_samples):
        information = 0.0
        for curr_class_num_samples in class_index_num_samples:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_valid_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _calculate_potential_information(values_num_samples, num_valid_samples):
        partition_potential_information = 0.0
        for value_num_samples in values_num_samples:
            if value_num_samples != 0:
                curr_ratio = value_num_samples / num_valid_samples
                partition_potential_information -= curr_ratio * math.log2(curr_ratio)
        return partition_potential_information


    @staticmethod
    def _generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
        # TESTED!
        random_classes = np.random.choice(len(classes_dist),
                                          num_valid_samples,
                                          replace=True,
                                          p=classes_dist)
        random_contingency_table = np.zeros((values_num_samples.shape[0], len(classes_dist)),
                                            dtype=float)
        samples_done = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples > 0:
                for class_index in random_classes[samples_done: samples_done + value_num_samples]:
                    random_contingency_table[value, class_index] += 1
                samples_done += value_num_samples
        return random_contingency_table

    @classmethod
    def _accept_attribute(cls, real_gain_ratio, num_tests, num_fails_allowed, num_valid_samples,
                          class_index_num_samples, values_num_samples):
        num_classes = len(class_index_num_samples)
        classes_dist = class_index_num_samples[:]
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)

        num_fails_seen = 0
        for test_number in range(1, num_tests + 1):
            random_contingency_table = cls._generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                values_num_samples)
            new_class_index_num_samples = np.sum(random_contingency_table, axis=0).tolist()

            original_information = cls._calculate_information(new_class_index_num_samples,
                                                              num_valid_samples)
            curr_gain_ratio = cls._calculate_gain_ratio(
                num_valid_samples,
                random_contingency_table,
                values_num_samples,
                original_information)
            if curr_gain_ratio > real_gain_ratio:
                num_fails_seen += 1
                if num_fails_seen > num_fails_allowed:
                    return False, test_number
            if num_tests - test_number <= num_fails_allowed - num_fails_seen:
                return True, test_number
        return True, num_tests



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                   INFORMATION GAIN                                        ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class InformationGain(Criterion):
    """Information Gain criterion. For reference see "Quinlan, J. R. C4.5: Programs for Machine
    Learning. Morgan Kaufmann Publishers, 1993.".
    """
    name = 'Information Gain'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the Information Gain
        criterion, using `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times. If `num_tests` is zero, returns the attribute/split with the
        largest criterion value.
        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.
        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """

        # First we calculate the original class frequency and information
        original_information = cls._calculate_information(tree_node.class_index_num_samples,
                                                          len(tree_node.valid_samples_indices))
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_information_gain = cls._calculate_information_gain(
                    len(tree_node.valid_samples_indices),
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    original_information)
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_information_gain))

        if num_tests == 0: # Just return attribute/split with maximum Information Gain.
            if best_splits_per_attrib:
                best_split = max(best_splits_per_attrib, key=lambda split: split.criterion_value)
            else:
                best_split = Split()
            num_monte_carlo_tests_needed = 0
            position_of_accepted = 1
            return (best_split, num_monte_carlo_tests_needed, position_of_accepted)
        else: # use Monte Carlo approach.
            if ORDER_RANDOMLY:
                random.shuffle(best_splits_per_attrib)
            else:
                best_splits_per_attrib.sort(key=lambda split: -split.criterion_value)
                if USE_ONE_ATTRIB_PER_NUM_VALUES:
                    best_splits_per_attrib_clean = []
                    num_values_seen = set()
                    for curr_split in best_splits_per_attrib:
                        num_values = len(curr_split.splits_values)
                        if num_values not in num_values_seen:
                            num_values_seen.add(num_values)
                            best_splits_per_attrib_clean.append(curr_split)
                    best_splits_per_attrib = best_splits_per_attrib_clean

            total_num_tests_needed = 0
            for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
                (should_accept,
                 num_tests_needed) = cls._accept_attribute(
                     best_attrib_split.criterion_value,
                     num_tests,
                     num_fails_allowed,
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.contingency_tables[
                         best_attrib_split.attrib_index].values_num_samples)
                total_num_tests_needed += num_tests_needed
                if should_accept:
                    return (best_attrib_split, total_num_tests_needed, curr_position + 1)
            return (Split(), total_num_tests_needed, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_information_gain(cls, num_valid_samples, contingency_table, values_num_samples,
                                    original_information):
        information_gain = original_information # Initial Information Gain
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            curr_split_information = cls._calculate_information(contingency_table[value],
                                                                value_num_samples)
            information_gain -= (value_num_samples / num_valid_samples) * curr_split_information
        return information_gain

    @staticmethod
    def _calculate_information(class_index_num_samples, num_valid_samples):
        information = 0.0
        for curr_class_num_samples in class_index_num_samples:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_valid_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
        # TESTED!
        random_classes = np.random.choice(len(classes_dist),
                                          num_valid_samples,
                                          replace=True,
                                          p=classes_dist)
        random_contingency_table = np.zeros((values_num_samples.shape[0], len(classes_dist)),
                                            dtype=float)
        samples_done = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples > 0:
                for class_index in random_classes[samples_done: samples_done + value_num_samples]:
                    random_contingency_table[value, class_index] += 1
                samples_done += value_num_samples
        return random_contingency_table

    @classmethod
    def _accept_attribute(cls, real_information_gain, num_tests, num_fails_allowed,
                          num_valid_samples, class_index_num_samples, values_num_samples):
        num_classes = len(class_index_num_samples)
        classes_dist = class_index_num_samples[:]
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)

        num_fails_seen = 0
        for test_number in range(1, num_tests + 1):
            random_contingency_table = cls._generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                values_num_samples)
            new_class_index_num_samples = np.sum(random_contingency_table, axis=0).tolist()

            original_information = cls._calculate_information(new_class_index_num_samples,
                                                              num_valid_samples)
            curr_information_gain = cls._calculate_information_gain(
                num_valid_samples,
                random_contingency_table,
                values_num_samples,
                original_information)
            if curr_information_gain > real_information_gain:
                num_fails_seen += 1
                if num_fails_seen > num_fails_allowed:
                    return False, test_number
            if num_tests - test_number <= num_fails_allowed - num_fails_seen:
                return True, test_number
        return True, num_tests



#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################




#################################################################################################
#################################################################################################
###                                                                                           ###
###                           CONDITIONAL INFERENCE TREE MULTIWAY                             ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class ConditionalInferenceTreeMultiway(Criterion):
    """
    Conditional Inference Tree using a multiway split. For reference, see "Unbiased Recursive
    Partitioning: A Conditional Inference Framework, T. Hothorn, K. Hornik & A. Zeileis. Journal of
    Computational and Graphical Statistics Vol. 15 , Iss. 3,2006".
    """
    name = 'Conditional Inference Tree Multiway'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, using the Conditional Inference Tree
        Framework to choose the best attribute and doing a multiway split on it. Note that this
        criterion does NOT accept the Monte Carlo Framework: it will raise a ValueError if
        `num_tests` > 0.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Should always be zero, otherwise will throw a `ValueError`.
            Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.

        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_c_quad_cdf = cls._calculate_c_quad_cdf(
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    tree_node.class_index_num_samples,
                    len(tree_node.valid_samples_indices))
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_c_quad_cdf))
        if num_tests == 0: # Just return attribute/split with maximum c_quad cdf.
            if best_splits_per_attrib:
                best_split = max(best_splits_per_attrib, key=lambda split: split.criterion_value)
            else:
                best_split = Split()
            num_monte_carlo_tests_needed = 0
            position_of_accepted = 1
            return (best_split, num_monte_carlo_tests_needed, position_of_accepted)
        else: # use Monte Carlo approach.
            raise ValueError

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_c_quad_cdf(cls, contingency_table, values_num_samples, class_index_num_samples,
                              num_valid_samples):
        def _calculate_expected_value_h(class_index_num_samples, num_valid_samples):
            return (1./num_valid_samples) * np.array(class_index_num_samples)

        def _calculate_covariance_h(expected_value_h, class_index_num_samples, num_valid_samples):
            num_classes = len(class_index_num_samples)
            covariance_h = np.zeros((num_classes, num_classes))
            for class_index, class_num_samples in enumerate(class_index_num_samples):
                if class_num_samples:
                    curr_class_one_hot_encoding = np.zeros((num_classes))
                    curr_class_one_hot_encoding[class_index] = 1.
                    diff = curr_class_one_hot_encoding - expected_value_h
                    covariance_h += class_num_samples * np.outer(diff, diff)
            return covariance_h / num_valid_samples

        def _calculate_mu_j(values_num_samples, expected_value_h):
            return np.outer(values_num_samples, expected_value_h).flatten(order='F')

        def _calculate_sigma_j(values_num_samples, num_valid_samples, covariance_h):
            values_num_samples_correct_dim = values_num_samples.reshape(
                (values_num_samples.shape[0], 1))
            return (((num_valid_samples / (num_valid_samples - 1))
                     * np.kron(covariance_h, np.diag(values_num_samples)))
                    - ((1 / (num_valid_samples - 1))
                       * np.kron(covariance_h,
                                 np.kron(values_num_samples_correct_dim,
                                         values_num_samples_correct_dim.transpose()))))


        expected_value_h = _calculate_expected_value_h(class_index_num_samples, num_valid_samples)
        covariance_h = _calculate_covariance_h(expected_value_h,
                                               class_index_num_samples,
                                               num_valid_samples)
        mu_j = _calculate_mu_j(values_num_samples, expected_value_h)
        sigma_j = _calculate_sigma_j(values_num_samples, num_valid_samples, covariance_h)

        temp_diff = contingency_table.flatten(order='F') - mu_j

        curr_rcond = 1e-15
        while True:
            try:
                sigma_j_pinv = np.linalg.pinv(sigma_j)
                sigma_j_rank = np.linalg.matrix_rank(sigma_j)
                break
            except np.linalg.linalg.LinAlgError:
                # Happens when sigma_j is (very) badly conditioned
                pass
            try:
                (sigma_j_pinv, sigma_j_rank) = scipy.linalg.pinv(sigma_j, return_rank=True)
                break
            except:
                # Happens when sigma_j is (very) badly conditioned
                curr_rcond *= 10.
                if curr_rcond > 1e-6:
                    # We give up on this attribute
                    print('Warning: attribute has sigma_j matrix that is not decomposable in SVD.')
                    return float('-inf')

        c_quad = np.dot(temp_diff, np.dot(sigma_j_pinv, temp_diff.transpose()))
        return scipy.stats.chi2.cdf(x=c_quad, df=sigma_j_rank)



#################################################################################################
#################################################################################################
###                                                                                           ###
###                            CONDITIONAL INFERENCE TREE TWOING                              ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class ConditionalInferenceTreeTwoing(Criterion):
    """
    Conditional Inference Tree using Twoing criterion to find best split. For references, see
    "Unbiased Recursive Partitioning: A Conditional Inference Framework, T. Hothorn, K. Hornik & A.
    Zeileis. Journal of Computational and Graphical Statistics Vol. 15 , Iss. 3,2006" and
    "Breiman, L., Friedman, J. J., Olshen, R. A., and Stone, C. J. Classification and Regression
    Trees. Wadsworth, 1984".
    """
    name = 'Conditional Inference Tree Twoing'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, using the Conditional Inference Tree
        Framework to choose the best attribute and using the Twoing criterion to find the best split
        for it. Note that this criterion does NOT accept the Monte Carlo Framework: it will raise a
        ValueError if `num_tests` > 0.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Should always be zero, otherwise will throw a `ValueError`.
            Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.

        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                curr_c_quad_cdf = cls._calculate_c_quad_cdf(
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    tree_node.class_index_num_samples,
                    len(tree_node.valid_samples_indices))
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=[], # will be calculated later
                                                    criterion_value=curr_c_quad_cdf))
        if num_tests == 0: # Just return attribute/split with maximum c_quad cdf.
            if best_splits_per_attrib:
                best_split = max(best_splits_per_attrib, key=lambda split: split.criterion_value)
                # Let's find the best split for this attribute using the Twoing criterion.
                best_total_gini_gain = float('-inf')
                best_left_values = set()
                best_right_values = set()
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[best_split.attrib_index].values_num_samples)
                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[best_split.attrib_index].contingency_table,
                         tree_node.contingency_tables[best_split.attrib_index].values_num_samples,
                         set_left_classes,
                         set_right_classes)
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    (curr_gini_gain,
                     left_values,
                     right_values) = cls._two_class_trick(
                         original_gini,
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[best_split.attrib_index].values_num_samples,
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    if curr_gini_gain > best_total_gini_gain:
                        best_total_gini_gain = curr_gini_gain
                        best_left_values = left_values
                        best_right_values = right_values
                best_split = Split(attrib_index=best_split.attrib_index,
                                   splits_values=[best_left_values, best_right_values],
                                   criterion_value=best_split.criterion_value)
            else:
                best_split = Split()
            num_monte_carlo_tests_needed = 0
            position_of_accepted = 1
            return (best_split, num_monte_carlo_tests_needed, position_of_accepted)
        else: # use Monte Carlo approach.
            raise ValueError

    @classmethod
    def _calculate_c_quad_cdf(cls, contingency_table, values_num_samples, class_index_num_samples,
                              num_valid_samples):
        def _calculate_expected_value_h(class_index_num_samples, num_valid_samples):
            return (1./num_valid_samples) * np.array(class_index_num_samples)

        def _calculate_covariance_h(expected_value_h, class_index_num_samples, num_valid_samples):
            num_classes = len(class_index_num_samples)
            covariance_h = np.zeros((num_classes, num_classes))
            for class_index, class_num_samples in enumerate(class_index_num_samples):
                if class_num_samples:
                    curr_class_one_hot_encoding = np.zeros((num_classes))
                    curr_class_one_hot_encoding[class_index] = 1.
                    diff = curr_class_one_hot_encoding - expected_value_h
                    covariance_h += class_num_samples * np.outer(diff, diff)
            return covariance_h / num_valid_samples

        def _calculate_mu_j(values_num_samples, expected_value_h):
            return np.outer(values_num_samples, expected_value_h).flatten(order='F')

        def _calculate_sigma_j(values_num_samples, num_valid_samples, covariance_h):
            values_num_samples_correct_dim = values_num_samples.reshape(
                (values_num_samples.shape[0], 1))
            return (((num_valid_samples / (num_valid_samples - 1))
                     * np.kron(covariance_h, np.diag(values_num_samples)))
                    - ((1 / (num_valid_samples - 1))
                       * np.kron(covariance_h,
                                 np.kron(values_num_samples_correct_dim,
                                         values_num_samples_correct_dim.transpose()))))


        expected_value_h = _calculate_expected_value_h(class_index_num_samples, num_valid_samples)
        covariance_h = _calculate_covariance_h(expected_value_h,
                                               class_index_num_samples,
                                               num_valid_samples)
        mu_j = _calculate_mu_j(values_num_samples, expected_value_h)
        sigma_j = _calculate_sigma_j(values_num_samples, num_valid_samples, covariance_h)

        temp_diff = contingency_table.flatten(order='F') - mu_j

        curr_rcond = 1e-15
        while True:
            try:
                sigma_j_pinv = np.linalg.pinv(sigma_j)
                sigma_j_rank = np.linalg.matrix_rank(sigma_j)
                break
            except np.linalg.linalg.LinAlgError:
                # Happens when sigma_j is (very) badly conditioned
                pass
            try:
                (sigma_j_pinv, sigma_j_rank) = scipy.linalg.pinv(sigma_j, return_rank=True)
                break
            except:
                # Happens when sigma_j is (very) badly conditioned
                curr_rcond *= 10.
                if curr_rcond > 1e-6:
                    # We give up on this attribute
                    print('Warning: attribute has sigma_j matrix that is not decomposable in SVD.')
                    return float('-inf')

        c_quad = np.dot(temp_diff, np.dot(sigma_j_pinv, temp_diff.transpose()))
        return scipy.stats.chi2.cdf(x=c_quad, df=sigma_j_rank)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_twoing(class_index_num_samples):
        # We only need to look at superclasses of up to (len(class_index_num_samples)/2 + 1)
        # elements because of symmetry! The subsets we are not choosing are complements of the ones
        # chosen.
        non_empty_classes = set([])
        for class_index, class_num_samples in enumerate(class_index_num_samples):
            if class_num_samples > 0:
                non_empty_classes.add(class_index)
        number_non_empty_classes = len(non_empty_classes)

        for left_classes in itertools.chain.from_iterable(
                itertools.combinations(non_empty_classes, size_left_superclass)
                for size_left_superclass in range(1, number_non_empty_classes // 2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if not set_left_classes or not set_right_classes:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_twoing_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                      set_right_classes):
        twoing_contingency_table = np.zeros((contingency_table.shape[0], 2), dtype=float)
        superclass_index_num_samples = [0, 0]
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index in set_left_classes:
                superclass_index_num_samples[0] += contingency_table[value][class_index]
                twoing_contingency_table[value][0] += contingency_table[value][class_index]
            for class_index in set_right_classes:
                superclass_index_num_samples[1] += contingency_table[value][class_index]
                twoing_contingency_table[value][1] += contingency_table[value][class_index]
        return twoing_contingency_table, superclass_index_num_samples

    @staticmethod
    def _two_class_trick(original_gini, class_index_num_samples, values_seen, values_num_samples,
                         contingency_table, num_total_valid_samples):
        # TESTED!
        def _get_non_empty_class_indices(class_index_num_samples):
            # TESTED!
            first_non_empty_class = None
            second_non_empty_class = None
            for class_index, class_num_samples in enumerate(class_index_num_samples):
                if class_num_samples > 0:
                    if first_non_empty_class is None:
                        first_non_empty_class = class_index
                    else:
                        second_non_empty_class = class_index
                        break
            return first_non_empty_class, second_non_empty_class

        def _calculate_value_class_ratio(values_seen, values_num_samples, contingency_table,
                                         non_empty_class_indices):
            # TESTED!
            value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = contingency_table[curr_value][second_class_index]
                value_number_ratio.append((curr_value,
                                           number_second_non_empty,
                                           number_second_non_empty/values_num_samples[curr_value]))
            value_number_ratio.sort(key=lambda tup: tup[2])
            return value_number_ratio

        def _calculate_children_gini_index(num_left_first, num_left_second, num_right_first,
                                           num_right_second, num_left_samples, num_right_samples):
            # TESTED!
            if num_left_samples != 0:
                left_first_class_freq_ratio = float(num_left_first)/float(num_left_samples)
                left_second_class_freq_ratio = float(num_left_second)/float(num_left_samples)
                left_split_gini_index = (1.0
                                         - left_first_class_freq_ratio**2
                                         - left_second_class_freq_ratio**2)
            else:
                # We can set left_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                left_split_gini_index = 1.0

            if num_right_samples != 0:
                right_first_class_freq_ratio = float(num_right_first)/float(num_right_samples)
                right_second_class_freq_ratio = float(num_right_second)/float(num_right_samples)
                right_split_gini_index = (1.0
                                          - right_first_class_freq_ratio**2
                                          - right_second_class_freq_ratio**2)
            else:
                # We can set right_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                right_split_gini_index = 1.0

            curr_children_gini_index = ((num_left_samples * left_split_gini_index
                                         + num_right_samples * right_split_gini_index)
                                        / (num_left_samples + num_right_samples))
            return curr_children_gini_index

        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_class,
         second_non_empty_class) = _get_non_empty_class_indices(class_index_num_samples)
        if first_non_empty_class is None or second_non_empty_class is None:
            return (float('-inf'), {0}, set())

        value_number_ratio = _calculate_value_class_ratio(values_seen,
                                                          values_num_samples,
                                                          contingency_table,
                                                          (first_non_empty_class,
                                                           second_non_empty_class))

        best_split_total_gini_gain = float('-inf')
        best_last_left_index = 0

        num_left_first = 0
        num_left_second = 0
        num_left_samples = 0
        num_right_first = class_index_num_samples[first_non_empty_class]
        num_right_second = class_index_num_samples[second_non_empty_class]
        num_right_samples = num_total_valid_samples

        for last_left_index, (last_left_value, last_left_num_second, _) in enumerate(
                value_number_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            last_left_num_first = num_samples_last_left_value - last_left_num_second

            num_left_samples += num_samples_last_left_value
            num_left_first += last_left_num_first
            num_left_second += last_left_num_second
            num_right_samples -= num_samples_last_left_value
            num_right_first -= last_left_num_first
            num_right_second -= last_left_num_second

            curr_children_gini_index = _calculate_children_gini_index(num_left_first,
                                                                      num_left_second,
                                                                      num_right_first,
                                                                      num_right_second,
                                                                      num_left_samples,
                                                                      num_right_samples)
            curr_gini_gain = original_gini - curr_children_gini_index
            if curr_gini_gain > best_split_total_gini_gain:
                best_split_total_gini_gain = curr_gini_gain
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_gain, set_left_values, set_right_values)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side/side_num)**2
        return gini_index

    @classmethod
    def _calculate_children_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        children_gini_index = ((left_num * left_split_gini_index
                                + right_num * right_split_gini_index)
                               / (left_num + right_num))
        return children_gini_index



#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################




#################################################################################################
#################################################################################################
###                                                                                           ###
###                                   GINI GAIN MONTE CARLO                                   ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GiniGainMonteCarlo(Criterion):
    """Gini Gain criterion. For reference see "Breiman, L., Friedman, J. J., Olshen, R. A., and
    Stone, C. J. Classification and Regression Trees. Wadsworth, 1984".
    """
    name = 'Gini Gain Monte Carlo'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the Gini Gain criterion,
        using `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times. If `num_tests` is zero, returns the attribute/split with
        the largest criterion value. Note that this criteria gives a multiway-split.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.

        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """
        original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                  tree_node.class_index_num_samples)
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_children_gini_index = cls._calculate_children_gini_index(
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    len(tree_node.valid_samples_indices),)
                curr_total_gini_gain = original_gini - curr_children_gini_index
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_total_gini_gain))


        best_splits_per_attrib.sort(key=lambda split: -split.criterion_value)
        p_values_list = []
        for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
            curr_p_value = cls._calculate_p_value(
                best_attrib_split.criterion_value,
                len(tree_node.valid_samples_indices),
                tree_node.class_index_num_samples,
                tree_node.contingency_tables[best_attrib_split.attrib_index].values_num_samples)
            p_values_list.append((curr_position + 1, best_attrib_split, curr_p_value))
        global LAST_P_VALUES
        LAST_P_VALUES = p_values_list

        total_num_tests_needed = len(best_splits_per_attrib) * NUM_MONTE_CARLO_SIMULATIONS
        if not p_values_list:
            return (Split(), total_num_tests_needed, None)
        best_position, best_attrib_split, _ = min(p_values_list, key=lambda x: x[2])
        return (best_attrib_split, total_num_tests_needed, best_position + 1)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_children_gini_index(cls, contingency_table, values_num_seen, num_valid_samples):
        total_children_gini = 0.0
        for value_index, value_num_samples in enumerate(values_num_seen):
            if value_num_samples == 0:
                continue
            curr_child_gini_index = cls._calculate_gini_index(value_num_samples,
                                                              contingency_table[value_index, :])
            total_children_gini += (value_num_samples / num_valid_samples) * curr_child_gini_index
        return total_children_gini

    @staticmethod
    def _calculate_gini_index(num_samples, class_num_samples):
        gini_index = 1.0
        for curr_class_num_samples in class_num_samples:
            if curr_class_num_samples > 0:
                gini_index -= (curr_class_num_samples / num_samples)**2
        return gini_index

    @staticmethod
    def _generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
        # TESTED!
        random_classes = np.random.choice(len(classes_dist),
                                          num_valid_samples,
                                          replace=True,
                                          p=classes_dist)
        random_contingency_table = np.zeros((values_num_samples.shape[0], len(classes_dist)),
                                            dtype=float)
        samples_done = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples > 0:
                for class_index in random_classes[samples_done: samples_done + value_num_samples]:
                    random_contingency_table[value, class_index] += 1
                samples_done += value_num_samples
        return random_contingency_table

    @classmethod
    def _calculate_p_value(cls, real_gini_gain, num_valid_samples, class_index_num_samples,
                           values_num_samples):
        num_classes = len(class_index_num_samples)
        classes_dist = class_index_num_samples[:]
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)

        num_fails_seen = 0
        for _ in range(NUM_MONTE_CARLO_SIMULATIONS):
            random_contingency_table = cls._generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                values_num_samples)

            new_class_index_num_samples = np.sum(random_contingency_table, axis=0).tolist()
            father_gini_index = cls._calculate_gini_index(num_valid_samples,
                                                          new_class_index_num_samples)
            curr_children_gini_index = cls._calculate_children_gini_index(
                random_contingency_table,
                values_num_samples,
                num_valid_samples)
            curr_total_gini_gain = father_gini_index - curr_children_gini_index
            if curr_total_gini_gain > real_gini_gain:
                num_fails_seen += 1

        return num_fails_seen / NUM_MONTE_CARLO_SIMULATIONS



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                  TWOING MONTE CARLO                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class TwoingMonteCarlo(Criterion):
    """Twoing criterion. For reference see "Breiman, L., Friedman, J. J., Olshen, R. A., and
    Stone, C. J. Classification and Regression Trees. Wadsworth, 1984".
    """
    name = 'Twoing Monte Carlo'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the Twoing criterion, using
        `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times. If `num_tests` is zero, returns the attribute/split with the
        largest criterion value.
        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.
        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """
        best_splits_per_attrib = []
        values_seen_per_attrib = []
        for attrib_index, is_valid_nominal_attrib in enumerate(tree_node.valid_nominal_attribute):
            if not is_valid_nominal_attrib:
                values_seen_per_attrib.append(None)
                continue
            else:
                best_total_gini_gain = float('-inf')
                best_left_values = set()
                best_right_values = set()
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                values_seen_per_attrib.append(values_seen)
                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index].contingency_table,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         set_left_classes,
                         set_right_classes)
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    (curr_gini_gain,
                     left_values,
                     right_values) = cls._two_class_trick(
                         original_gini,
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    if curr_gini_gain > best_total_gini_gain:
                        best_total_gini_gain = curr_gini_gain
                        best_left_values = left_values
                        best_right_values = right_values
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=[best_left_values,
                                                                   best_right_values],
                                                    criterion_value=best_total_gini_gain))

        best_splits_per_attrib.sort(key=lambda split: -split.criterion_value)
        p_values_list = []
        for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
            curr_p_value = cls._calculate_p_value(
                best_attrib_split.criterion_value,
                len(tree_node.valid_samples_indices),
                tree_node.class_index_num_samples,
                tree_node.contingency_tables[best_attrib_split.attrib_index].values_num_samples,
                values_seen_per_attrib[best_attrib_split.attrib_index])
            p_values_list.append((curr_position + 1, best_attrib_split, curr_p_value))
        global LAST_P_VALUES
        LAST_P_VALUES = p_values_list

        total_num_tests_needed = len(best_splits_per_attrib) * NUM_MONTE_CARLO_SIMULATIONS
        if not p_values_list:
            return (Split(), total_num_tests_needed, None)
        best_position, best_attrib_split, _ = min(p_values_list, key=lambda x: x[2])
        return (best_attrib_split, total_num_tests_needed, best_position + 1)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_twoing(class_index_num_samples):
        # We only need to look at superclasses of up to (len(class_index_num_samples)/2 + 1)
        # elements because of symmetry! The subsets we are not choosing are complements of the ones
        # chosen.
        non_empty_classes = set([])
        for class_index, class_num_samples in enumerate(class_index_num_samples):
            if class_num_samples > 0:
                non_empty_classes.add(class_index)
        number_non_empty_classes = len(non_empty_classes)

        for left_classes in itertools.chain.from_iterable(
                itertools.combinations(non_empty_classes, size_left_superclass)
                for size_left_superclass in range(1, number_non_empty_classes // 2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if not set_left_classes or not set_right_classes:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_twoing_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                      set_right_classes):
        twoing_contingency_table = np.zeros((contingency_table.shape[0], 2), dtype=float)
        superclass_index_num_samples = [0, 0]
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index in set_left_classes:
                superclass_index_num_samples[0] += contingency_table[value][class_index]
                twoing_contingency_table[value][0] += contingency_table[value][class_index]
            for class_index in set_right_classes:
                superclass_index_num_samples[1] += contingency_table[value][class_index]
                twoing_contingency_table[value][1] += contingency_table[value][class_index]
        return twoing_contingency_table, superclass_index_num_samples

    @staticmethod
    def _two_class_trick(original_gini, class_index_num_samples, values_seen, values_num_samples,
                         contingency_table, num_total_valid_samples):
        # TESTED!
        def _get_non_empty_class_indices(class_index_num_samples):
            # TESTED!
            first_non_empty_class = None
            second_non_empty_class = None
            for class_index, class_num_samples in enumerate(class_index_num_samples):
                if class_num_samples > 0:
                    if first_non_empty_class is None:
                        first_non_empty_class = class_index
                    else:
                        second_non_empty_class = class_index
                        break
            return first_non_empty_class, second_non_empty_class

        def _calculate_value_class_ratio(values_seen, values_num_samples, contingency_table,
                                         non_empty_class_indices):
            # TESTED!
            value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = contingency_table[curr_value][second_class_index]
                value_number_ratio.append((curr_value,
                                           number_second_non_empty,
                                           number_second_non_empty/values_num_samples[curr_value]))
            value_number_ratio.sort(key=lambda tup: tup[2])
            return value_number_ratio

        def _calculate_children_gini_index(num_left_first, num_left_second, num_right_first,
                                           num_right_second, num_left_samples, num_right_samples):
            # TESTED!
            if num_left_samples != 0:
                left_first_class_freq_ratio = float(num_left_first)/float(num_left_samples)
                left_second_class_freq_ratio = float(num_left_second)/float(num_left_samples)
                left_split_gini_index = (1.0
                                         - left_first_class_freq_ratio**2
                                         - left_second_class_freq_ratio**2)
            else:
                # We can set left_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                left_split_gini_index = 1.0

            if num_right_samples != 0:
                right_first_class_freq_ratio = float(num_right_first)/float(num_right_samples)
                right_second_class_freq_ratio = float(num_right_second)/float(num_right_samples)
                right_split_gini_index = (1.0
                                          - right_first_class_freq_ratio**2
                                          - right_second_class_freq_ratio**2)
            else:
                # We can set right_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                right_split_gini_index = 1.0

            curr_children_gini_index = ((num_left_samples * left_split_gini_index
                                         + num_right_samples * right_split_gini_index)
                                        / (num_left_samples + num_right_samples))
            return curr_children_gini_index

        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_class,
         second_non_empty_class) = _get_non_empty_class_indices(class_index_num_samples)
        if first_non_empty_class is None or second_non_empty_class is None:
            return (float('-inf'), {0}, set())

        value_number_ratio = _calculate_value_class_ratio(values_seen,
                                                          values_num_samples,
                                                          contingency_table,
                                                          (first_non_empty_class,
                                                           second_non_empty_class))

        best_split_total_gini_gain = float('-inf')
        best_last_left_index = 0

        num_left_first = 0
        num_left_second = 0
        num_left_samples = 0
        num_right_first = class_index_num_samples[first_non_empty_class]
        num_right_second = class_index_num_samples[second_non_empty_class]
        num_right_samples = num_total_valid_samples

        for last_left_index, (last_left_value, last_left_num_second, _) in enumerate(
                value_number_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            last_left_num_first = num_samples_last_left_value - last_left_num_second

            num_left_samples += num_samples_last_left_value
            num_left_first += last_left_num_first
            num_left_second += last_left_num_second
            num_right_samples -= num_samples_last_left_value
            num_right_first -= last_left_num_first
            num_right_second -= last_left_num_second

            curr_children_gini_index = _calculate_children_gini_index(num_left_first,
                                                                      num_left_second,
                                                                      num_right_first,
                                                                      num_right_second,
                                                                      num_left_samples,
                                                                      num_right_samples)
            curr_gini_gain = original_gini - curr_children_gini_index
            if curr_gini_gain > best_split_total_gini_gain:
                best_split_total_gini_gain = curr_gini_gain
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_gain, set_left_values, set_right_values)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side/side_num)**2
        return gini_index

    @classmethod
    def _calculate_children_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        children_gini_index = ((left_num * left_split_gini_index
                                + right_num * right_split_gini_index)
                               / (left_num + right_num))
        return children_gini_index

    @staticmethod
    def _generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
        # TESTED!
        random_classes = np.random.choice(len(classes_dist),
                                          num_valid_samples,
                                          replace=True,
                                          p=classes_dist)
        random_contingency_table = np.zeros((values_num_samples.shape[0], len(classes_dist)),
                                            dtype=float)
        samples_done = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples > 0:
                for class_index in random_classes[samples_done: samples_done + value_num_samples]:
                    random_contingency_table[value, class_index] += 1
                samples_done += value_num_samples
        return random_contingency_table

    @classmethod
    def _calculate_p_value(cls, real_gini_gain, num_valid_samples, class_index_num_samples,
                           values_num_samples, values_seen):
        num_classes = len(class_index_num_samples)
        classes_dist = class_index_num_samples[:]
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)

        num_fails_seen = 0
        for _ in range(NUM_MONTE_CARLO_SIMULATIONS):
            random_contingency_table = cls._generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                values_num_samples)

            best_gini_gain = float('-inf')
            for (set_left_classes,
                 set_right_classes) in cls._generate_twoing(class_index_num_samples):

                (twoing_contingency_table,
                 superclass_index_num_samples) = cls._get_twoing_contingency_table(
                     random_contingency_table,
                     values_num_samples,
                     set_left_classes,
                     set_right_classes)
                original_gini = cls._calculate_gini_index(num_valid_samples,
                                                          superclass_index_num_samples)
                (curr_gini_gain, _, _) = cls._two_class_trick(original_gini,
                                                              superclass_index_num_samples,
                                                              values_seen,
                                                              values_num_samples,
                                                              twoing_contingency_table,
                                                              num_valid_samples)
                if curr_gini_gain > best_gini_gain:
                    best_gini_gain = curr_gini_gain
            if best_gini_gain > real_gini_gain:
                num_fails_seen += 1
        return num_fails_seen / NUM_MONTE_CARLO_SIMULATIONS



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                 GAIN RATIO MONTE CARLO                                    ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GainRatioMonteCarlo(Criterion):
    """Gain Ratio criterion. For reference see "Quinlan, J. R. C4.5: Programs for Machine Learning.
    Morgan Kaufmann Publishers, 1993.".
    """
    name = 'Gain Ratio Monte Carlo'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the Gain Ratio criterion,
        using `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times. If `num_tests` is zero, returns the attribute/split with the
        largest criterion value.
        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.
        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """

        # First we calculate the original class frequency and information
        original_information = cls._calculate_information(tree_node.class_index_num_samples,
                                                          len(tree_node.valid_samples_indices))
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_gain_ratio = cls._calculate_gain_ratio(
                    len(tree_node.valid_samples_indices),
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    original_information)
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_gain_ratio))

        best_splits_per_attrib.sort(key=lambda split: -split.criterion_value)
        p_values_list = []
        for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
            curr_p_value = cls._calculate_p_value(
                best_attrib_split.criterion_value,
                len(tree_node.valid_samples_indices),
                tree_node.class_index_num_samples,
                tree_node.contingency_tables[best_attrib_split.attrib_index].values_num_samples)
            p_values_list.append((curr_position + 1, best_attrib_split, curr_p_value))
        global LAST_P_VALUES
        LAST_P_VALUES = p_values_list

        total_num_tests_needed = len(best_splits_per_attrib) * NUM_MONTE_CARLO_SIMULATIONS
        if not p_values_list:
            return (Split(), total_num_tests_needed, None)
        best_position, best_attrib_split, _ = min(p_values_list, key=lambda x: x[2])
        return (best_attrib_split, total_num_tests_needed, best_position + 1)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_gain_ratio(cls, num_valid_samples, contingency_table, values_num_samples,
                              original_information):
        information_gain = original_information # Initial Information Gain
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            curr_split_information = cls._calculate_information(contingency_table[value],
                                                                value_num_samples)
            information_gain -= (value_num_samples / num_valid_samples) * curr_split_information

        # Gain Ratio
        potential_partition_information = cls._calculate_potential_information(values_num_samples,
                                                                               num_valid_samples)
        # Note that, since there are at least two different values, potential_partition_information
        # is never zero.
        gain_ratio = information_gain / potential_partition_information
        return gain_ratio

    @staticmethod
    def _calculate_information(class_index_num_samples, num_valid_samples):
        information = 0.0
        for curr_class_num_samples in class_index_num_samples:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_valid_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _calculate_potential_information(values_num_samples, num_valid_samples):
        partition_potential_information = 0.0
        for value_num_samples in values_num_samples:
            if value_num_samples != 0:
                curr_ratio = value_num_samples / num_valid_samples
                partition_potential_information -= curr_ratio * math.log2(curr_ratio)
        return partition_potential_information


    @staticmethod
    def _generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
        # TESTED!
        random_classes = np.random.choice(len(classes_dist),
                                          num_valid_samples,
                                          replace=True,
                                          p=classes_dist)
        random_contingency_table = np.zeros((values_num_samples.shape[0], len(classes_dist)),
                                            dtype=float)
        samples_done = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples > 0:
                for class_index in random_classes[samples_done: samples_done + value_num_samples]:
                    random_contingency_table[value, class_index] += 1
                samples_done += value_num_samples
        return random_contingency_table

    @classmethod
    def _calculate_p_value(cls, real_gain_ratio, num_valid_samples, class_index_num_samples,
                           values_num_samples):
        num_classes = len(class_index_num_samples)
        classes_dist = class_index_num_samples[:]
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)

        num_fails_seen = 0
        for _ in range(NUM_MONTE_CARLO_SIMULATIONS):
            random_contingency_table = cls._generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                values_num_samples)
            new_class_index_num_samples = np.sum(random_contingency_table, axis=0).tolist()

            original_information = cls._calculate_information(new_class_index_num_samples,
                                                              num_valid_samples)
            curr_gain_ratio = cls._calculate_gain_ratio(
                num_valid_samples,
                random_contingency_table,
                values_num_samples,
                original_information)
            if curr_gain_ratio > real_gain_ratio:
                num_fails_seen += 1
        return num_fails_seen / NUM_MONTE_CARLO_SIMULATIONS



#################################################################################################
#################################################################################################
###                                                                                           ###
###                           INFORMATION GAIN MONTE CARLO                                    ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class InformationGainMonteCarlo(Criterion):
    """Information Gain criterion. For reference see "Quinlan, J. R. C4.5: Programs for Machine
    Learning. Morgan Kaufmann Publishers, 1993.".
    """
    name = 'Information Gain Monte Carlo'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the Information Gain
        criterion, using `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times. If `num_tests` is zero, returns the attribute/split with the
        largest criterion value.
        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.
        Returns:
            A tuple cointaining, in order:
                - The best split found;
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """

        # First we calculate the original class frequency and information
        original_information = cls._calculate_information(tree_node.class_index_num_samples,
                                                          len(tree_node.valid_samples_indices))
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_information_gain = cls._calculate_information_gain(
                    len(tree_node.valid_samples_indices),
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    original_information)
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_information_gain))
        best_splits_per_attrib.sort(key=lambda split: -split.criterion_value)
        p_values_list = []
        for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
            curr_p_value = cls._calculate_p_value(
                best_attrib_split.criterion_value,
                len(tree_node.valid_samples_indices),
                tree_node.class_index_num_samples,
                tree_node.contingency_tables[best_attrib_split.attrib_index].values_num_samples)
            p_values_list.append((curr_position + 1, best_attrib_split, curr_p_value))
        global LAST_P_VALUES
        LAST_P_VALUES = p_values_list

        total_num_tests_needed = len(best_splits_per_attrib) * NUM_MONTE_CARLO_SIMULATIONS
        if not p_values_list:
            return (Split(), total_num_tests_needed, None)
        best_position, best_attrib_split, _ = min(p_values_list, key=lambda x: x[2])
        return (best_attrib_split, total_num_tests_needed, best_position + 1)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_information_gain(cls, num_valid_samples, contingency_table, values_num_samples,
                                    original_information):
        information_gain = original_information # Initial Information Gain
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            curr_split_information = cls._calculate_information(contingency_table[value],
                                                                value_num_samples)
            information_gain -= (value_num_samples / num_valid_samples) * curr_split_information
        return information_gain

    @staticmethod
    def _calculate_information(class_index_num_samples, num_valid_samples):
        information = 0.0
        for curr_class_num_samples in class_index_num_samples:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_valid_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
        # TESTED!
        random_classes = np.random.choice(len(classes_dist),
                                          num_valid_samples,
                                          replace=True,
                                          p=classes_dist)
        random_contingency_table = np.zeros((values_num_samples.shape[0], len(classes_dist)),
                                            dtype=float)
        samples_done = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples > 0:
                for class_index in random_classes[samples_done: samples_done + value_num_samples]:
                    random_contingency_table[value, class_index] += 1
                samples_done += value_num_samples
        return random_contingency_table

    @classmethod
    def _calculate_p_value(cls, real_information_gain, num_valid_samples, class_index_num_samples,
                           values_num_samples):
        num_classes = len(class_index_num_samples)
        classes_dist = class_index_num_samples[:]
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)

        num_fails_seen = 0
        for _ in range(NUM_MONTE_CARLO_SIMULATIONS):
            random_contingency_table = cls._generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                values_num_samples)
            new_class_index_num_samples = np.sum(random_contingency_table, axis=0).tolist()

            original_information = cls._calculate_information(new_class_index_num_samples,
                                                              num_valid_samples)
            curr_information_gain = cls._calculate_information_gain(
                num_valid_samples,
                random_contingency_table,
                values_num_samples,
                original_information)
            if curr_information_gain > real_information_gain:
                num_fails_seen += 1
        return num_fails_seen / NUM_MONTE_CARLO_SIMULATIONS
