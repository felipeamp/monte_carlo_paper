#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing all criteria available for tests.


"""

import abc
import itertools
import math
import random

import numpy as np


#: Maximum "time complexity" allowed when generating an attribute best split.
LIMIT_EXPONENTIAL_STEPS = 3000000
#: Maximum log2 of "time complexity" allowed when generating an attribute best split.
LOG2_LIMIT_EXPONENTIAL_STEPS = 16
#: Whether Monte Carlo Framework should order attributes randomly or in decreasing criterion value.
ORDER_RANDOMLY = False

class Criterion(object):
    __metaclass__ = abc.ABCMeta

    name = ''

    @classmethod
    @abc.abstractmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the criterion, using
        `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times.
        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.
        """
        # returns (separation_attrib_index, splits_values, criterion_value, num_tests_needed,
        #          position_of_accepted)
        pass



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                        GINI GAIN                                          ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GiniGain(Criterion):
    name = 'Gini Gain'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        """Returns the best attribute and its best split, according to the Gini Gain criterion,
        using `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times. If `num_tests` is zero, returns the attribute/split with
        the largest criterion value.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.

        Returns:
            A tuple cointaining, in order:
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """
        original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                  tree_node.class_index_num_samples)
        best_splits_per_attrib = []
        has_exactly_two_classes = tree_node.number_non_empty_classes == 2
        cache_values_seen = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if not is_valid_attrib:
                cache_values_seen.append(None)
                continue
            else:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index][1])
                cache_values_seen.append(values_seen)
                if (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                        (tree_node.number_non_empty_classes
                         * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                if has_exactly_two_classes:
                    (curr_total_gini_gain,
                     left_values,
                     right_values) = cls._two_class_trick(
                         original_gini,
                         tree_node.class_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         tree_node.contingency_tables[attrib_index][0],
                         len(tree_node.valid_samples_indices))
                    best_splits_per_attrib.append((attrib_index
                                                   [left_values, right_values],
                                                   curr_total_gini_gain))
                else:
                    best_total_gini_gain = float('-inf')
                    best_left_values = set()
                    best_right_values = set()
                    for (left_values,
                         right_values,
                         left_num,
                         class_num_left,
                         right_num,
                         class_num_right) in cls._generate_possible_splits(
                             tree_node.contingency_tables[attrib_index][1],
                             values_seen,
                             tree_node.contingency_tables[attrib_index][0],
                             tree_node.dataset.num_classes):
                        curr_children_gini_index = cls._calculate_children_gini_index(
                            left_num,
                            class_num_left,
                            right_num,
                            class_num_right)
                        curr_total_gini_gain = original_gini - curr_children_gini_index
                        if curr_total_gini_gain > best_total_gini_gain:
                            best_total_gini_gain = curr_total_gini_gain
                            best_left_values = left_values
                            best_right_values = right_values
                    best_splits_per_attrib.append((attrib_index,
                                                   [best_left_values,
                                                    best_right_values],
                                                   best_total_gini_gain))
        if num_tests == 0: # Just return attribute/split with maximum Gini Gain.
            best_attribute_and_split = (None, [], float('-inf'))
            for curr_attrib_split in best_splits_per_attrib:
                if curr_attrib_split[2] > best_attribute_and_split[2]:
                    best_attribute_and_split = curr_attrib_split
            num_monte_carlo_tests_needed = 0
            position_of_accepted = 1
            return (*best_attribute_and_split,
                    num_monte_carlo_tests_needed,
                    position_of_accepted)
        else: # use Monte Carlo approach.
            if ORDER_RANDOMLY:
                random.shuffle(best_splits_per_attrib)
            else:
                best_splits_per_attrib.sort(key=lambda x: -x[2])

            total_num_tests_needed = 0
            for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
                attrib_index, _, criterion_value = best_attrib_split
                (should_accept,
                 num_tests_needed) = cls._accept_attribute(
                     original_gini,
                     criterion_value,
                     num_tests,
                     num_fails_allowed,
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.contingency_tables[attrib_index][1],
                     has_exactly_two_classes,
                     cache_values_seen[attrib_index])
                total_num_tests_needed += num_tests_needed
                if should_accept:
                    return (*best_attrib_split, total_num_tests_needed, curr_position + 1)
            return (None, [], float('-inf'), total_num_tests_needed, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

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
                value_number_ratio.append(
                    (curr_value,
                     number_second_non_empty,
                     number_second_non_empty / values_num_samples[curr_value]))
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
            curr_total_gini_gain = original_gini - curr_children_gini_index
            if curr_total_gini_gain > best_split_total_gini_gain:
                best_split_total_gini_gain = curr_total_gini_gain
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_gain, set_left_values, set_right_values)

    @staticmethod
    def _generate_possible_splits(values_num_samples, values_seen, contingency_table,
                                  num_classes):
        # TESTED!
        # We only need to look at subsets of up to (len(values_seen)/2 + 1) elements because of
        # symmetry! The subsets we are not choosing are complements of the ones chosen.
        for left_values in itertools.chain.from_iterable(
                itertools.combinations(values_seen, size_left_side)
                for size_left_side in range(len(values_seen)//2 + 1)):
            set_left_values = set(left_values)
            set_right_values = values_seen - set_left_values

            left_num = 0
            class_num_left = [0] * num_classes
            right_num = 0
            class_num_right = [0] * num_classes
            for value in set_left_values:
                left_num += values_num_samples[value]
                for class_index in range(num_classes):
                    class_num_left[class_index] += contingency_table[value][class_index]
            for value in set_right_values:
                right_num += values_num_samples[value]
                for class_index in range(num_classes):
                    class_num_right[class_index] += contingency_table[value][class_index]

            if left_num == 0 or right_num == 0:
                # A valid split must have at least one sample in each side
                continue
            yield (set_left_values, set_right_values, left_num, class_num_left, right_num,
                   class_num_right)

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
    def _accept_attribute(cls, original_gini, real_gini_gain, num_tests, num_fails_allowed,
                          num_valid_samples, class_index_num_samples, values_num_samples,
                          has_exactly_two_classes, values_seen):
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

            if has_exactly_two_classes:
                (best_gini_gain_found, _, _) = cls._two_class_trick(
                    original_gini,
                    class_index_num_samples,
                    values_seen,
                    values_num_samples,
                    random_contingency_table,
                    num_valid_samples)
            else:
                best_gini_gain_found = float('-inf')
                for (_, _, left_num,
                     class_num_left,
                     right_num,
                     class_num_right) in cls._generate_possible_splits(
                         values_num_samples,
                         values_seen,
                         random_contingency_table,
                         num_classes):
                    curr_children_gini_index = cls._calculate_children_gini_index(
                        left_num,
                        class_num_left,
                        right_num,
                        class_num_right)
                    curr_total_gini_gain = original_gini - curr_children_gini_index
                    if curr_total_gini_gain > best_gini_gain_found:
                        best_gini_gain_found = curr_total_gini_gain

            if best_gini_gain_found > real_gini_gain:
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
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
                - Total number of Monte Carlo tests needed;
                - Position of the accepted attribute in the attributes' list ordered by the
                criterion value.
        """
        best_splits_per_attrib = []
        cache_values_seen = []
        for attrib_index, is_valid_nominal_attrib in enumerate(tree_node.valid_nominal_attribute):
            if not is_valid_nominal_attrib:
                cache_values_seen.append(None)
                continue
            else:
                best_total_gini_gain = float('-inf')
                best_left_values = set()
                best_right_values = set()
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index][1])
                cache_values_seen.append(values_seen)
                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.contingency_tables[attrib_index][1],
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
                         tree_node.contingency_tables[attrib_index][1],
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    if curr_gini_gain > best_total_gini_gain:
                        best_total_gini_gain = curr_gini_gain
                        best_left_values = left_values
                        best_right_values = right_values
                best_splits_per_attrib.append((attrib_index,
                                               [best_left_values, best_right_values],
                                               best_total_gini_gain))
        if num_tests == 0: # Just return attribute/split with maximum criterion value.
            best_attribute_and_split = (None, [], float('-inf'))
            for curr_attrib_split in best_splits_per_attrib:
                if curr_attrib_split[2] > best_attribute_and_split[2]:
                    best_attribute_and_split = curr_attrib_split
            num_monte_carlo_tests_needed = 0
            position_of_accepted = 1
            return (*best_attribute_and_split,
                    num_monte_carlo_tests_needed,
                    position_of_accepted)
        else: # use Monte Carlo approach.
            if ORDER_RANDOMLY:
                random.shuffle(best_splits_per_attrib)
            else:
                best_splits_per_attrib.sort(key=lambda x: -x[2])

            total_num_tests_needed = 0
            for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
                attrib_index, _, criterion_value = best_attrib_split
                (should_accept,
                 num_tests_needed) = cls._accept_attribute(
                     criterion_value,
                     num_tests,
                     num_fails_allowed,
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.contingency_tables[attrib_index][1],
                     cache_values_seen[attrib_index])
                total_num_tests_needed += num_tests_needed
                if should_accept:
                    return (*best_attrib_split, total_num_tests_needed, curr_position + 1)
            return (None, [], float('-inf'), total_num_tests_needed, None)

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
                - the index of the accepted attribute;
                - a list of sets, each containing the values that should go to that split/subtree.
                -  Split value according to the criterion. If no attribute has a valid split, this
                value should be `float('-inf')`.
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
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                splits_values = [set([value]) for value in values_seen]
                curr_gain_ratio = cls._calculate_gain_ratio(
                    len(tree_node.valid_samples_indices),
                    tree_node.contingency_tables[attrib_index][0],
                    tree_node.contingency_tables[attrib_index][1],
                    original_information)
                best_splits_per_attrib.append((attrib_index,
                                               splits_values,
                                               curr_gain_ratio))

        if num_tests == 0: # Just return attribute/split with maximum criterion value.
            best_attribute_and_split = (None, [], float('-inf'))
            for curr_attrib_split in best_splits_per_attrib:
                if curr_attrib_split[2] > best_attribute_and_split[2]:
                    best_attribute_and_split = curr_attrib_split
            num_monte_carlo_tests_needed = 0
            position_of_accepted = 1
            return (*best_attribute_and_split,
                    num_monte_carlo_tests_needed,
                    position_of_accepted)
        else: # use Monte Carlo approach.
            if ORDER_RANDOMLY:
                random.shuffle(best_splits_per_attrib)
            else:
                best_splits_per_attrib.sort(key=lambda x: -x[2])

            total_num_tests_needed = 0
            for curr_position, best_attrib_split in enumerate(best_splits_per_attrib):
                attrib_index, _, criterion_value = best_attrib_split
                (should_accept,
                 num_tests_needed) = cls._accept_attribute(
                     criterion_value,
                     num_tests,
                     num_fails_allowed,
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.contingency_tables[attrib_index][1])
                total_num_tests_needed += num_tests_needed
                if should_accept:
                    return (*best_attrib_split, total_num_tests_needed, curr_position + 1)
            return (None, [], float('-inf'), total_num_tests_needed, None)

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
