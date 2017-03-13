#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module containing all criteria available for tests.
'''

import abc
import itertools
import math
import random
import sys
import timeit

import cvxpy as cvx
import numpy as np
from scipy.stats import chi2

import chol


LIMIT_EXPONENTIAL_STEPS = 50000000
LOG2_LIMIT_EXPONENTIAL_STEPS = 25
NUM_TESTS_CHI_SQUARE_MONTE_CARLO = 30

class Criterion(object):
    __metaclass__ = abc.ABCMeta

    name = ''

    @classmethod
    @abc.abstractmethod
    def select_best_attribute_and_split(cls, tree_node, num_tests=0, num_fails_allowed=0):
        '''Returns the best attribute and its best split, according to the criterion, using
        `num_tests` tests per attribute and accepting if it doesn't fail more than
        `num_fails_allowed` times.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.
          num_tests (int, optional): number of tests to be executed in each attribute, according to
            our Monte Carlo framework. Defaults to `0`.
          num_fails_allowed (int, optional): maximum number of fails allowed for an attribute to be
            accepted according to our Monte Carlo framework. Defaults to `0`.
        '''
        # returns (separation_attrib_index, splits_values, criterion_value, p_value)
        pass



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       GINI INDEX                                          ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GiniIndex(Criterion):
    #TESTED!
    name = 'Gini Index'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        # MODIFIED!
        #TESTED!
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)
        original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                  tree_node.class_index_num_samples)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                if tree_node.number_non_empty_classes == 2:
                    (curr_total_gini_index,
                     left_values,
                     right_values) = cls._two_class_trick(
                         tree_node.class_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         tree_node.contingency_tables[attrib_index][0],
                         len(tree_node.valid_samples_indices))
                    curr_gini = (original_gini
                                 - curr_total_gini_index/len(tree_node.valid_samples_indices))
                    ret.append((attrib_index,
                                curr_gini,
                                [left_values, right_values],
                                None,
                                timeit.default_timer() - start_time,
                                None,
                                None))
                else:
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
                        curr_total_gini_index = cls._calculate_total_gini_index(
                            left_num,
                            class_num_left,
                            right_num,
                            class_num_right)
                        curr_gini = (original_gini
                                     - curr_total_gini_index/len(tree_node.valid_samples_indices))
                        ret.append((attrib_index,
                                    curr_gini,
                                    [left_values, right_values],
                                    None,
                                    timeit.default_timer() - start_time,
                                    None,
                                    None))
        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!

        # Instead of using original_total_gini_index = (
        # gini_index(all_samples_in_tree_node)
        # - ((len(samples_in_left_node)/len(all_samples_in_tree_node))
        #    * gini_index(samples_in_left_node))
        # - ((len(samples_in_right_node)/len(all_samples_in_tree_node))
        #    * gini_index(samples_in_right_node))
        # and trying to maximize it, we'll try to minimize
        # total_gini_index = (len(samples_in_left_node) * gini_index(samples_in_left_node)
        #                     + len(samples_in_left_node) * gini_index(samples_in_right_node))

        # Since total_gini_index above is always non-negative and <= 2 (because gini_index is always
        # non-negative and <= 1.0), the starting value below will be replaced in the for loop.
        best_split_total_gini_index = float('inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                if tree_node.number_non_empty_classes == 2:
                    (curr_total_gini_index,
                     left_values,
                     right_values) = cls._two_class_trick(
                         tree_node.class_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         tree_node.contingency_tables[attrib_index][0],
                         len(tree_node.valid_samples_indices))
                    if curr_total_gini_index < best_split_total_gini_index:
                        best_split_total_gini_index = curr_total_gini_index
                        best_split_attrib_index = attrib_index
                        best_split_left_values = left_values
                        best_split_right_values = right_values
                else:
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
                        curr_total_gini_index = cls._calculate_total_gini_index(
                            left_num,
                            class_num_left,
                            right_num,
                            class_num_right)
                        if curr_total_gini_index < best_split_total_gini_index:
                            best_split_total_gini_index = curr_total_gini_index
                            best_split_attrib_index = attrib_index
                            best_split_left_values = left_values
                            best_split_right_values = right_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_total_gini_index, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _two_class_trick(class_index_num_samples, values_seen, values_num_samples,
                         contingency_table, num_total_valid_samples):
        # MODIFIED!
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
            # MODIFIED!
            # TESTED!
            value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = contingency_table[curr_value][second_class_index]
                value_number_ratio.append((curr_value,
                                           number_second_non_empty,
                                           number_second_non_empty/values_num_samples[curr_value]))
            value_number_ratio = sorted(value_number_ratio, key=lambda tup: tup[2])
            return value_number_ratio

        def _calculate_gini_index(num_left_first, num_left_second, num_right_first,
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
                # by zero in curr_total_gini_index
                left_split_gini_index = 1.0

            if num_right_samples != 0:
                right_first_class_freq_ratio = float(num_right_first)/float(num_right_samples)
                right_second_class_freq_ratio = float(num_right_second)/float(num_right_samples)
                right_split_gini_index = (1.0
                                          - right_first_class_freq_ratio**2
                                          - right_second_class_freq_ratio**2)
            else:
                # We can set right_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_total_gini_index
                right_split_gini_index = 1.0

            curr_total_gini_index = (num_left_samples * left_split_gini_index
                                     + num_right_samples * right_split_gini_index)
            return curr_total_gini_index

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

        # Since total_gini_index above is always non-negative and <= 2 (because gini_index is
        # always non-negative and <= 1.0), the starting value below will be replaced in the for
        # loop.
        best_split_total_gini_index = float('inf') # > 2.0
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

            curr_total_gini_index = _calculate_gini_index(num_left_first,
                                                          num_left_second,
                                                          num_right_first,
                                                          num_right_second,
                                                          num_left_samples,
                                                          num_right_samples)
            if curr_total_gini_index < best_split_total_gini_index:
                best_split_total_gini_index = curr_total_gini_index
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_index, set_left_values, set_right_values)

    @staticmethod
    def _generate_possible_splits(values_num_samples, values_seen, contingency_table, num_classes):
        # MODIFIED!
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
        # MODIFIED!
        #TESTED!
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side/side_num)**2
        return gini_index

    @classmethod
    def _calculate_total_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        # MODIFIED!
        #TESTED!
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        total_gini_index = left_num * left_split_gini_index + right_num * right_split_gini_index
        return total_gini_index



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       GINI TWOING                                         ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GiniTwoing(Criterion):
    name = 'Gini Twoing'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                best_total_gini_index = float('-inf')
                best_left_values = set()
                best_right_values = set()

                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.contingency_tables[attrib_index][1],
                         set_left_classes,
                         set_right_classes)
                    (curr_total_gini_index,
                     left_values,
                     right_values) = cls._two_class_trick(
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))

                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    curr_gini = (original_gini
                                 - curr_total_gini_index/len(tree_node.valid_samples_indices))
                    if curr_gini > best_total_gini_index:
                        best_total_gini_index = curr_gini
                        best_left_values = left_values
                        best_right_values = right_values

                twoing_value = cls._get_twoing_value_for_nominal_optimum(
                    tree_node.contingency_tables[attrib_index][0],
                    tree_node.contingency_tables[attrib_index][1],
                    tree_node.dataset.num_classes,
                    best_left_values,
                    best_right_values)
                ret.append((attrib_index,
                            twoing_value,
                            [best_left_values, best_right_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

            elif is_valid_numeric_attrib:
                start_time = timeit.default_timer()
                (values_seen,
                 values_and_classes) = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                    tree_node.dataset.samples,
                                                                    tree_node.dataset.sample_class,
                                                                    attrib_index)
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                sorted_values_and_classes = sorted(values_and_classes)
                (best_twoing,
                 last_left_value,
                 first_right_value) = cls._twoing_for_numeric(
                     sorted_values_and_classes,
                     tree_node.dataset.num_classes)
                ret.append((attrib_index,
                            best_twoing,
                            [{last_left_value}, {first_right_value}],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!
        best_split_total_twoing = float('-inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                best_twoing_curr_attrib = float('-inf')
                best_split_left_values_curr_attrib = set()
                best_split_right_values_curr_attrib = set()
                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.contingency_tables[attrib_index][1],
                         set_left_classes,
                         set_right_classes)
                    (curr_total_gini_index,
                     left_values,
                     right_values) = cls._two_class_trick(
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    curr_gini = (original_gini
                                 - curr_total_gini_index/len(tree_node.valid_samples_indices))
                    if curr_gini > best_twoing_curr_attrib:
                        best_twoing_curr_attrib = curr_gini
                        best_split_left_values_curr_attrib = left_values
                        best_split_right_values_curr_attrib = right_values

                twoing_value = cls._get_twoing_value_for_nominal_optimum(
                    tree_node.contingency_tables[attrib_index][0],
                    tree_node.contingency_tables[attrib_index][1],
                    tree_node.dataset.num_classes,
                    best_split_left_values_curr_attrib,
                    best_split_right_values_curr_attrib)
                if twoing_value > best_split_total_twoing:
                    best_split_total_twoing = twoing_value
                    best_split_attrib_index = attrib_index
                    best_split_left_values = best_split_left_values_curr_attrib
                    best_split_right_values = best_split_right_values_curr_attrib

            elif is_valid_numeric_attrib:
                (values_seen,
                 values_and_classes) = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                    tree_node.dataset.samples,
                                                                    tree_node.dataset.sample_class,
                                                                    attrib_index)
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                sorted_values_and_classes = sorted(values_and_classes)
                (best_twoing_curr_attrib,
                 last_left_value,
                 first_right_value) = cls._twoing_for_numeric(
                     sorted_values_and_classes,
                     tree_node.dataset.num_classes)
                if best_twoing_curr_attrib > best_split_total_twoing:
                    best_split_total_twoing = best_twoing_curr_attrib
                    best_split_attrib_index = attrib_index
                    best_split_left_values = {last_left_value}
                    best_split_right_values = {first_right_value}

        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_total_twoing, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_seen = set()
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
            if sample_value not in values_seen:
                values_seen.add(sample_value)
        return values_seen, values_and_classes

    @staticmethod
    def _get_sorted_values_and_twoing_classes(sorted_values_and_classes, set_left_classes,
                                              set_right_classes):
        sorted_values_and_twoing_classes = []
        for (value, sample_class) in sorted_values_and_classes:
            if sample_class in set_left_classes:
                sorted_values_and_twoing_classes.append((value, 0))
            elif sample_class in set_right_classes:
                sorted_values_and_twoing_classes.append((value, 1))
            else:
                print('Sample class ({}) not contained in twoing!'.format(sample_class))
                print('set_left_classes:', set_left_classes)
                print('set_right_classes:', set_right_classes)
                sys.exit(1)
        return sorted_values_and_twoing_classes

    @staticmethod
    def _generate_twoing(class_index_num_samples):
        #TESTED!

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
                for size_left_superclass in range(1, number_non_empty_classes//2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if len(set_left_classes) == 0 or len(set_right_classes) == 0:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_twoing_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                      set_right_classes):
        # MODIFIED!
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
    def _get_twoing_value(class_num_left, class_num_right, num_left_samples,
                          num_right_samples):
        sum_dif = 0.0
        for left_num, right_num in zip(class_num_left, class_num_right):
            class_num_tot = class_num_left + class_num_right
            if class_num_tot == 0:
                continue
            sum_dif += abs(left_num / num_left_samples - right_num / num_right_samples)

        num_total_samples = num_left_samples + num_right_samples
        frequency_left = num_left_samples / num_total_samples
        frequency_right = num_right_samples / num_total_samples

        twoing_value = (frequency_left * frequency_right / 4.0) * sum_dif ** 2
        return twoing_value

    @classmethod
    def _get_twoing_value_for_nominal_optimum(cls, contingency_table, values_num_samples,
                                              num_classes, set_left_values, set_right_values):
        class_num_left = [0] * num_classes
        class_num_right = [0] * num_classes
        num_left_samples = 0
        num_right_samples = 0
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            if value in set_left_values:
                num_left_samples += value_num_samples
                for (class_index,
                     value_class_num_samples) in enumerate(contingency_table[value]):
                    if value_class_num_samples > 0:
                        class_num_left[class_index] += value_class_num_samples
            elif value in set_right_values:
                num_right_samples += value_num_samples
                for (class_index,
                     value_class_num_samples) in enumerate(contingency_table[value]):
                    if value_class_num_samples > 0:
                        class_num_right[class_index] += value_class_num_samples
            else:
                print('Unkown value in split:', value)
                print('set_left_values:', set_left_values)
                print('set_right_values:', set_right_values)
                sys.exit(1)
        return  cls._get_twoing_value(class_num_left,
                                      class_num_right,
                                      num_left_samples,
                                      num_right_samples)

    @classmethod
    def _twoing_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_twoing = float('-inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                twoing_value = cls._get_twoing_value(class_num_left,
                                                     class_num_right,
                                                     num_left_samples,
                                                     num_right_samples)
                if twoing_value > best_twoing:
                    best_twoing = twoing_value
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_twoing, best_last_left_value, best_first_right_value)

    @staticmethod
    def _two_class_trick(class_index_num_samples, values_seen, values_num_samples,
                         contingency_table, num_total_valid_samples):
        # MODIFIED!
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
            # MODIFIED!
            # TESTED!
            value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = contingency_table[curr_value][second_class_index]
                value_number_ratio.append((curr_value,
                                           number_second_non_empty,
                                           number_second_non_empty/values_num_samples[curr_value]))
            value_number_ratio = sorted(value_number_ratio, key=lambda tup: tup[2])
            return value_number_ratio

        def _calculate_gini_index(num_left_first, num_left_second, num_right_first,
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
                # by zero in curr_total_gini_index
                left_split_gini_index = 1.0

            if num_right_samples != 0:
                right_first_class_freq_ratio = float(num_right_first)/float(num_right_samples)
                right_second_class_freq_ratio = float(num_right_second)/float(num_right_samples)
                right_split_gini_index = (1.0
                                          - right_first_class_freq_ratio**2
                                          - right_second_class_freq_ratio**2)
            else:
                # We can set right_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_total_gini_index
                right_split_gini_index = 1.0

            curr_total_gini_index = (num_left_samples * left_split_gini_index
                                     + num_right_samples * right_split_gini_index)
            return curr_total_gini_index

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

        best_split_total_gini_index = float('inf')
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

            curr_total_gini_index = _calculate_gini_index(num_left_first,
                                                          num_left_second,
                                                          num_right_first,
                                                          num_right_second,
                                                          num_left_samples,
                                                          num_right_samples)
            if curr_total_gini_index < best_split_total_gini_index:
                best_split_total_gini_index = curr_total_gini_index
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_index, set_left_values, set_right_values)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        # MODIFIED!
        #TESTED!
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side/side_num)**2
        return gini_index

    @classmethod
    def _calculate_total_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        # MODIFIED!
        #TESTED!
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        total_gini_index = left_num * left_split_gini_index + right_num * right_split_gini_index
        return total_gini_index






#################################################################################################
#################################################################################################
###                                                                                           ###
###                                 GINI TWOING MONTE CARLO                                   ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GiniTwoingMonteCarlo(Criterion):
    name = 'Gini Twoing Monte Carlo'


    @classmethod
    def evaluate_all_attributes_2(cls, tree_node, num_tests, num_fails_allowed):
        # contains (attrib_index, gini, split_values, p_value, time_taken)
        best_split_per_attrib = []

        num_valid_attrib = 0
        # num_tests = int(math.ceil(math.log2(num_valid_attrib))) + 6
        values_seen_per_attrib = {}
        criterion_start_time = timeit.default_timer()
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                values_seen_per_attrib[attrib_index] = values_seen
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                num_valid_attrib += 1

                best_total_gini_index = float('-inf')
                best_left_values = set()
                best_right_values = set()

                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.contingency_tables[attrib_index][1],
                         set_left_classes,
                         set_right_classes)
                    (curr_total_gini_index,
                     left_values,
                     right_values) = cls._two_class_trick(
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    curr_gini = (original_gini
                                 - curr_total_gini_index/len(tree_node.valid_samples_indices))

                    if curr_gini > best_total_gini_index:
                        best_total_gini_index = curr_gini
                        best_left_values = left_values
                        best_right_values = right_values

                best_split_per_attrib.append((attrib_index,
                                              best_total_gini_index,
                                              [best_left_values, best_right_values],
                                              None,
                                              timeit.default_timer() - start_time))
        criterion_total_time = timeit.default_timer() - criterion_start_time

        # Order splits by gini value
        ordered_start_time = timeit.default_timer()
        preference_rank_full = sorted(best_split_per_attrib, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []#(1,
        #                     float('-inf'),
        #                     None,
        #                     None,
        #                     None)]
        # bad_attrib_indices = {3, 5, 6, 10, 11, 12, 13, 17, 18, 20, 21, 22, 25, 56, 57, 52, 55, 59,
        #                       60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 476, 478}
        # preference_rank_mailcode_first = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
            # if pref_elem[0] in bad_attrib_indices:
            #     preference_rank_mailcode_first.append(pref_elem)

        # preference_rank_mailcode_first = sorted(preference_rank_mailcode_first,
        #                                         key=lambda x: x[1])
        # for pref_elem in preference_rank:
        #     if pref_elem[0] in bad_attrib_indices:
        #         continue
        #     preference_rank_mailcode_first.append(pref_elem)

        tests_done_ordered = 0
        accepted_attribute_ordered = None
        ordered_accepted_rank = None
        for (rank_index,
             (attrib_index, best_total_gini_index, _, _, _)) in enumerate(preference_rank):
            if math.isinf(best_total_gini_index):
                continue
            values_seen = values_seen_per_attrib[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_total_gini_index,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 tree_node.class_index_num_samples,
                 tree_node.contingency_tables[attrib_index][0],
                 tree_node.contingency_tables[attrib_index][1],
                 values_seen,
                 num_fails_allowed)
            if not should_accept:
                tests_done_ordered += num_tests_needed
            else:
                accepted_attribute_ordered = tree_node.dataset.attrib_names[attrib_index]
                ordered_accepted_rank = rank_index + 1
                print('Accepted attribute:', accepted_attribute_ordered)
                tests_done_ordered += num_tests
                break
        ordered_total_time = timeit.default_timer() - ordered_start_time


        rev_start_time = timeit.default_timer()
        # Reversed ordered
        rev_preference_rank = reversed(preference_rank)

        tests_done_rev = 0
        accepted_attribute_rev = None
        for (attrib_index, best_total_gini_index, _, _, _) in rev_preference_rank:
            if math.isinf(best_total_gini_index):
                continue
            values_seen = values_seen_per_attrib[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_total_gini_index,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 tree_node.class_index_num_samples,
                 tree_node.contingency_tables[attrib_index][0],
                 tree_node.contingency_tables[attrib_index][1],
                 values_seen,
                 num_fails_allowed)
            if not should_accept:
                tests_done_rev += num_tests_needed
            else:
                accepted_attribute_rev = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_rev)
                tests_done_rev += num_tests
                break
        rev_total_time = timeit.default_timer() - rev_start_time

        # Order splits randomly
        random_start_time = timeit.default_timer()
        random_order_rank = preference_rank[:]
        random.shuffle(random_order_rank)

        tests_done_random_order = 0
        accepted_attribute_random = None
        for (attrib_index, best_total_gini_index, _, _, _) in random_order_rank:
            if math.isinf(best_total_gini_index):
                continue
            values_seen = values_seen_per_attrib[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_total_gini_index,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 tree_node.class_index_num_samples,
                 tree_node.contingency_tables[attrib_index][0],
                 tree_node.contingency_tables[attrib_index][1],
                 values_seen,
                 num_fails_allowed)
            if not should_accept:
                tests_done_random_order += num_tests_needed
            else:
                accepted_attribute_random = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_random)
                tests_done_random_order += num_tests
                break
        random_total_time = timeit.default_timer() - random_start_time

        if ordered_accepted_rank is None:
            return (tests_done_ordered,
                    accepted_attribute_ordered,
                    tests_done_rev,
                    accepted_attribute_rev,
                    tests_done_random_order,
                    accepted_attribute_random,
                    num_valid_attrib,
                    ordered_accepted_rank,
                    criterion_total_time,
                    ordered_total_time,
                    rev_total_time,
                    random_total_time,
                    preference_rank[0],
                    None)
        else:
            return (tests_done_ordered,
                    accepted_attribute_ordered,
                    tests_done_rev,
                    accepted_attribute_rev,
                    tests_done_random_order,
                    accepted_attribute_random,
                    num_valid_attrib,
                    ordered_accepted_rank,
                    criterion_total_time,
                    ordered_total_time,
                    rev_total_time,
                    random_total_time,
                    preference_rank[0],
                    preference_rank[ordered_accepted_rank - 1])

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)

        num_valid_attrib = sum(tree_node.valid_nominal_attribute)
        num_tests = int(math.ceil(math.log2(num_valid_attrib))) + 6

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                best_total_gini_index = float('-inf')
                best_left_values = set()
                best_right_values = set()

                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.contingency_tables[attrib_index][1],
                         set_left_classes,
                         set_right_classes)
                    (curr_total_gini_index,
                     left_values,
                     right_values) = cls._two_class_trick(
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    curr_gini = (original_gini
                                 - curr_total_gini_index/len(tree_node.valid_samples_indices))

                    if curr_gini > best_total_gini_index:
                        best_total_gini_index = curr_gini
                        best_left_values = left_values
                        best_right_values = right_values

                (should_accept,
                 num_tests_needed) = cls.accept_attribute(
                     best_total_gini_index,
                     num_tests,
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1],
                     values_seen,
                     num_fails_allowed=0)
                ret.append((attrib_index,
                            best_total_gini_index,
                            [best_left_values, best_right_values],
                            None,
                            timeit.default_timer() - start_time,
                            should_accept,
                            num_tests_needed))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!
        best_split_total_gini_index = float('-inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        num_valid_attrib = sum(tree_node.valid_nominal_attribute)
        num_tests = int(math.ceil(math.log2(num_valid_attrib))) + 6

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                best_attrib_gini_index = float('-inf')
                best_attrib_left_values = set()
                best_attrib_right_values = set()

                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.contingency_tables[attrib_index][1],
                         set_left_classes,
                         set_right_classes)
                    (curr_total_gini_index,
                     left_values,
                     right_values) = cls._two_class_trick(
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index][1],
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    curr_gini = (original_gini
                                 - curr_total_gini_index/len(tree_node.valid_samples_indices))

                    if curr_gini > best_attrib_gini_index:
                        best_attrib_gini_index = curr_gini
                        best_attrib_left_values = left_values
                        best_attrib_right_values = right_values

                (should_accept, _) = cls.accept_attribute(
                    best_attrib_gini_index,
                    num_tests,
                    len(tree_node.valid_samples_indices),
                    tree_node.class_index_num_samples,
                    tree_node.contingency_tables[attrib_index][0],
                    tree_node.contingency_tables[attrib_index][1],
                    values_seen,
                    num_fails_allowed=0)
                if not should_accept:
                    continue
                elif best_attrib_gini_index > best_split_total_gini_index:
                    best_split_total_gini_index = best_attrib_gini_index
                    best_split_attrib_index = attrib_index
                    best_split_left_values = best_attrib_left_values
                    best_split_right_values = best_attrib_right_values

        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_total_gini_index, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_twoing(class_index_num_samples):
        #TESTED!

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
                for size_left_superclass in range(1, number_non_empty_classes//2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if len(set_left_classes) == 0 or len(set_right_classes) == 0:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_twoing_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                      set_right_classes):
        # MODIFIED!
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
    def _two_class_trick(class_index_num_samples, values_seen, values_num_samples,
                         contingency_table, num_total_valid_samples):
        # MODIFIED!
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
            # MODIFIED!
            # TESTED!
            value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = contingency_table[curr_value][second_class_index]
                value_number_ratio.append((curr_value,
                                           number_second_non_empty,
                                           number_second_non_empty/values_num_samples[curr_value]))
            value_number_ratio = sorted(value_number_ratio, key=lambda tup: tup[2])
            return value_number_ratio

        def _calculate_gini_index(num_left_first, num_left_second, num_right_first,
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
                # by zero in curr_total_gini_index
                left_split_gini_index = 1.0

            if num_right_samples != 0:
                right_first_class_freq_ratio = float(num_right_first)/float(num_right_samples)
                right_second_class_freq_ratio = float(num_right_second)/float(num_right_samples)
                right_split_gini_index = (1.0
                                          - right_first_class_freq_ratio**2
                                          - right_second_class_freq_ratio**2)
            else:
                # We can set right_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_total_gini_index
                right_split_gini_index = 1.0

            curr_total_gini_index = (num_left_samples * left_split_gini_index
                                     + num_right_samples * right_split_gini_index)
            return curr_total_gini_index

        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_class,
         second_non_empty_class) = _get_non_empty_class_indices(class_index_num_samples)
        if first_non_empty_class is None or second_non_empty_class is None:
            return (float('inf'), {0}, set())

        value_number_ratio = _calculate_value_class_ratio(values_seen,
                                                          values_num_samples,
                                                          contingency_table,
                                                          (first_non_empty_class,
                                                           second_non_empty_class))

        best_split_total_gini_index = float('inf')
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

            curr_total_gini_index = _calculate_gini_index(num_left_first,
                                                          num_left_second,
                                                          num_right_first,
                                                          num_right_second,
                                                          num_left_samples,
                                                          num_right_samples)
            if curr_total_gini_index < best_split_total_gini_index:
                best_split_total_gini_index = curr_total_gini_index
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_index, set_left_values, set_right_values)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        # MODIFIED!
        #TESTED!
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side/side_num)**2
        return gini_index

    @classmethod
    def _calculate_total_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        # MODIFIED!
        #TESTED!
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        total_gini_index = left_num * left_split_gini_index + right_num * right_split_gini_index
        return total_gini_index

    @staticmethod
    def get_classes_dist(contingency_table, values_num_samples, num_valid_samples):
        num_classes = contingency_table.shape[1]
        classes_dist = [0] * num_classes
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index, num_samples in enumerate(contingency_table[value, :]):
                if num_samples > 0:
                    classes_dist[class_index] += num_samples
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)
        return classes_dist

    @staticmethod
    def generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
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
    def accept_attribute(cls, real_gini, num_tests, num_valid_samples, class_index_num_samples,
                         contingency_table, values_num_samples, values_seen, num_fails_allowed):
        classes_dist = cls.get_classes_dist(contingency_table,
                                            values_num_samples,
                                            num_valid_samples)
        num_fails_seen = 0
        for test_number in range(1, num_tests + 1):
            random_contingency_table = cls.generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                values_num_samples)

            best_gini_found = float('-inf')
            for (set_left_classes,
                 set_right_classes) in cls._generate_twoing(class_index_num_samples):

                (twoing_contingency_table,
                 superclass_index_num_samples) = cls._get_twoing_contingency_table(
                     random_contingency_table,
                     values_num_samples,
                     set_left_classes,
                     set_right_classes)
                (curr_total_gini_index, _, _) = cls._two_class_trick(superclass_index_num_samples,
                                                                     values_seen,
                                                                     values_num_samples,
                                                                     twoing_contingency_table,
                                                                     num_valid_samples)
                original_gini = cls._calculate_gini_index(num_valid_samples,
                                                          superclass_index_num_samples)
                curr_gini = (original_gini
                             - curr_total_gini_index / num_valid_samples)
                if curr_gini > best_gini_found:
                    best_gini_found = curr_gini

            if best_gini_found > real_gini:
                num_fails_seen += 1
                if num_fails_seen > num_fails_allowed:
                    return False, test_number
            if num_tests - test_number <= num_fails_allowed - num_fails_seen:
                return True, None
        return True, None



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                     TWOING CRITERION                                      ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class Twoing(Criterion):
    name = 'Twoing'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue

                best_total_twoing_value = float('-inf')
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

                    # (set_left_classes,
                    #  set_right_classes) = cls._get_superclass_partition(class_num_left,
                    #                                                     class_num_right)

                    curr_twoing_value = cls._get_twoing_value(class_num_left,
                                                              class_num_right,
                                                              left_num,
                                                              right_num)
                    if curr_twoing_value > best_total_twoing_value:
                        best_total_twoing_value = curr_twoing_value
                        best_left_values = left_values
                        best_right_values = right_values

                ret.append((attrib_index,
                            best_total_twoing_value,
                            [best_left_values, best_right_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!
        best_split_twoing_value = float('-inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
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

                    # (set_left_classes,
                    #  set_right_classes) = cls._get_superclass_partition(class_num_left,
                    #                                                     class_num_right)

                    curr_twoing_value = cls._get_twoing_value(class_num_left,
                                                              class_num_right,
                                                              left_num,
                                                              right_num)
                    if curr_twoing_value > best_split_twoing_value:
                        best_split_twoing_value = curr_twoing_value
                        best_split_attrib_index = attrib_index
                        best_split_left_values = left_values
                        best_split_right_values = right_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_twoing_value, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_possible_splits(values_num_samples, values_seen, contingency_table, num_classes):
        # MODIFIED!
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
    def _get_superclass_partition(class_num_left, class_num_right):
        set_left_classes = set()
        set_right_classes = set()
        for (class_index,
             (class_left,
              class_right)) in enumerate(zip(class_num_left, class_num_right)):
            if class_left >= class_right:
                set_left_classes.add(class_index)
            else:
                set_right_classes.add(class_index)
        return set_left_classes, set_right_classes

    @staticmethod
    def _get_twoing_value(class_num_left, class_num_right, num_left_samples,
                          num_right_samples):
        sum_dif = 0.0
        for left_num, right_num in zip(class_num_left, class_num_right):
            class_num_tot = class_num_left + class_num_right
            if class_num_tot == 0:
                continue
            sum_dif += abs(left_num / num_left_samples - right_num / num_right_samples)

        num_total_samples = num_left_samples + num_right_samples
        frequency_left = num_left_samples / num_total_samples
        frequency_right = num_right_samples / num_total_samples

        twoing_value = (frequency_left * frequency_right / 4.0) * sum_dif ** 2
        return twoing_value



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                          ORT                                              ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class Ort(Criterion):
    #TESTED!
    name = 'ORT'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        # MODIFIED!
        #TESTED!
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     class_num_left,
                     _,
                     class_num_right) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_ort = cls._calculate_ort(
                        class_num_left,
                        class_num_right)
                    ret.append((attrib_index,
                                curr_ort,
                                [left_values, right_values],
                                None,
                                timeit.default_timer() - start_time,
                                None,
                                None))
        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!
        best_split_ort = float('-inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     class_num_left,
                     _,
                     class_num_right) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_ort = cls._calculate_ort(
                        class_num_left,
                        class_num_right)
                    if curr_ort > best_split_ort:
                        best_split_ort = curr_ort
                        best_split_attrib_index = attrib_index
                        best_split_left_values = left_values
                        best_split_right_values = right_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_ort, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_possible_splits(values_num_samples, values_seen, contingency_table, num_classes):
        # MODIFIED!
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
    def _calculate_ort(class_num_left, class_num_right):
        # MODIFIED!
        #TESTED!
        theta = 0.0
        left_side_norm_sq = 0.0
        right_side_norm_sq = 0.0
        for num_left, num_right in zip(class_num_left, class_num_right):
            theta += num_left * num_right
            left_side_norm_sq += num_left ** 2
            right_side_norm_sq += num_right ** 2
        left_side_norm = math.sqrt(left_side_norm_sq)
        right_side_norm = math.sqrt(right_side_norm_sq)
        if left_side_norm == 0.0 or right_side_norm == 0.0:
            return float('-inf')
        theta /= left_side_norm * right_side_norm
        return 1.0 - theta



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                          MPI                                              ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class Mpi(Criterion):
    #TESTED!
    name = 'MPI'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        # MODIFIED!
        #TESTED!
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
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
                    curr_mpi = cls._calculate_mpi(left_num,
                                                  class_num_left,
                                                  right_num,
                                                  class_num_right)
                    ret.append((attrib_index,
                                curr_mpi,
                                [left_values, right_values],
                                None,
                                timeit.default_timer() - start_time,
                                None,
                                None))
        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!
        best_split_mpi = float('-inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
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
                    curr_mpi = cls._calculate_mpi(left_num,
                                                  class_num_left,
                                                  right_num,
                                                  class_num_right)
                    if curr_mpi > best_split_mpi:
                        best_split_mpi = curr_mpi
                        best_split_attrib_index = attrib_index
                        best_split_left_values = left_values
                        best_split_right_values = right_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_mpi, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_possible_splits(values_num_samples, values_seen, contingency_table, num_classes):
        # MODIFIED!
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
    def _calculate_mpi(left_num, class_num_left, right_num, class_num_right):
        # MODIFIED!
        # TESTED!
        num_total = left_num + right_num
        beta = left_num * right_num
        for class_left, class_right in zip(class_num_left, class_num_right):
            class_total_samples = class_left + class_right
            beta -= class_total_samples * class_left * class_right / num_total
        return beta / (num_total**2)



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                    MAX CUT EXACT                                          ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutExact(Criterion):
    #TESTED!
    name = 'Max Cut Exact'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        # MODIFIED!
        #TESTED!
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     _,
                     right_num,
                     class_num_right) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_cut = cls._calculate_cut(left_values,
                                                  right_num,
                                                  class_num_right,
                                                  tree_node.contingency_tables[attrib_index][0],
                                                  tree_node.contingency_tables[attrib_index][1])
                    ret.append((attrib_index,
                                curr_cut,
                                [left_values, right_values],
                                None,
                                timeit.default_timer() - start_time,
                                None,
                                None))
        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!
        best_split_cut = float('-inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     _,
                     right_num,
                     class_num_right) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_cut = float(cls._calculate_cut(
                        left_values,
                        right_num,
                        class_num_right,
                        tree_node.contingency_tables[attrib_index][0],
                        tree_node.contingency_tables[attrib_index][1]))
                    if curr_cut > best_split_cut:
                        best_split_cut = curr_cut
                        best_split_attrib_index = attrib_index
                        best_split_left_values = left_values
                        best_split_right_values = right_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_cut, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_possible_splits(values_num_samples, values_seen, contingency_table, num_classes):
        # MODIFIED!
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
    def _calculate_cut(left_values, right_num, class_num_right, contingency_table,
                       values_num_samples):
        # MODIFIED!
        #TESTED!
        cut_val = 0
        for value, num_samples in enumerate(values_num_samples):
            if num_samples == 0:
                continue
            if value in left_values:
                for class_index, class_num_other_side in enumerate(class_num_right):
                    cut_val += (contingency_table[value][class_index] *
                                (right_num - class_num_other_side))
        return cut_val



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                 MAX CUT EXACT RESIDUE                                     ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutExactResidue(Criterion):
    #TESTED!
    name = 'Max Cut Exact Residue' # Using full node-dataset class distribution

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        # MODIFIED!
        #TESTED!
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     _,
                     _,
                     _) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_cut = cls._calculate_cut(left_values,
                                                  right_values,
                                                  tree_node.contingency_tables[attrib_index][0],
                                                  tree_node.contingency_tables[attrib_index][1])
                    ret.append((attrib_index,
                                curr_cut,
                                [left_values, right_values],
                                None,
                                timeit.default_timer() - start_time,
                                None,
                                None))
        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!
        best_split_cut = float('-inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     _,
                     _,
                     _) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_cut = cls._calculate_cut(left_values,
                                                  right_values,
                                                  tree_node.contingency_tables[attrib_index][0],
                                                  tree_node.contingency_tables[attrib_index][1])
                    if curr_cut > best_split_cut:
                        best_split_cut = curr_cut
                        best_split_attrib_index = attrib_index
                        best_split_left_values = left_values
                        best_split_right_values = right_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_cut, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_possible_splits(values_num_samples, values_seen, contingency_table, num_classes):
        # MODIFIED!
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
    def _calculate_cut(left_values, right_values, contingency_table, values_num_samples):
        # MODIFIED!
        #TESTED!
        cut_val = 0.0
        num_classes = contingency_table.shape[1]
        for value_left in left_values:
            for value_right in right_values:
                for class_index in range(num_classes):

                    cut_val += (contingency_table[value_left][class_index] *
                                (values_num_samples[value_right]
                                 - contingency_table[value_right][class_index]))

                    # Let's subtract the average cut for this pair of values with this distribution
                    mixed_class_dist = np.add(contingency_table[value_left],
                                              contingency_table[value_right])

                    total_elems_value_pair = (values_num_samples[value_left]
                                              + values_num_samples[value_right])
                    left_frac = values_num_samples[value_left] / total_elems_value_pair
                    right_frac = values_num_samples[value_right] / total_elems_value_pair

                    cut_val -= (left_frac * mixed_class_dist[class_index] *
                                (values_num_samples[value_right]
                                 - right_frac * mixed_class_dist[class_index]))
                    if cut_val < 0.0:
                        print('='*90)
                        print('VALOR DE PESO: {} < 0'.format(cut_val))
                        print('='*90)
        return cut_val



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                 MAX CUT EXACT CHI SQUARE                                  ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutExactChiSquare(Criterion):
    #TESTED!
    name = 'Max Cut Exact Chi Square'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        # MODIFIED!
        #TESTED!
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     _,
                     _,
                     _) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_cut = cls._calculate_cut(left_values,
                                                  right_values,
                                                  tree_node.contingency_tables[attrib_index][0],
                                                  tree_node.contingency_tables[attrib_index][1])
                    ret.append((attrib_index,
                                curr_cut,
                                [left_values, right_values],
                                None,
                                timeit.default_timer() - start_time,
                                None,
                                None))
        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!
        best_split_cut = float('-inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     _,
                     _,
                     _) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_cut = cls._calculate_cut(left_values,
                                                  right_values,
                                                  tree_node.contingency_tables[attrib_index][0],
                                                  tree_node.contingency_tables[attrib_index][1])
                    if curr_cut > best_split_cut:
                        best_split_cut = curr_cut
                        best_split_attrib_index = attrib_index
                        best_split_left_values = left_values
                        best_split_right_values = right_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_cut, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_possible_splits(values_num_samples, values_seen, contingency_table, num_classes):
        # MODIFIED!
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
    def _calculate_cut(left_values, right_values, contingency_table, values_num_samples):
        # MODIFIED!
        #TESTED!
        cut_val = 0.0
        num_classes = contingency_table.shape[1]
        for value_left in left_values:
            if values_num_samples[value_left] == 0:
                continue
            for value_right in right_values:
                if values_num_samples[value_right] == 0:
                    continue

                num_samples_both_values = (values_num_samples[value_left]
                                           + values_num_samples[value_right])
                for class_index in range(num_classes):
                    num_samples_both_values_this_class = (
                        contingency_table[value_left][class_index]
                        + contingency_table[value_right][class_index])
                    if num_samples_both_values_this_class == 0:
                        continue
                    expected_value_left_class = (
                        values_num_samples[value_left] * num_samples_both_values_this_class
                        / num_samples_both_values)
                    expected_value_right_class = (
                        values_num_samples[value_right] * num_samples_both_values_this_class
                        / num_samples_both_values)
                    diff_value_left = (contingency_table[value_left][class_index]
                                       - expected_value_left_class)
                    diff_value_right = (contingency_table[value_right][class_index]
                                        - expected_value_right_class)
                    cut_val += (
                        diff_value_left * (diff_value_left / expected_value_left_class)
                        + diff_value_right * (diff_value_right / expected_value_right_class))
                    if cut_val < 0.0:
                        print('='*90)
                        print('VALOR DE PESO: {} < 0'.format(cut_val))
                        print('='*90)
        return cut_val



#################################################################################################
#################################################################################################
###                                                                                           ###
###                            MAX CUT EXACT CHI SQUARE HEURISTIC                             ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutExactChiSquareHeuristic(Criterion):
    #TESTED!
    name = 'Max Cut Exact Chi Square Heuristic'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        # MODIFIED!
        #TESTED!
        ret = [] # contains (attrib_index, gini, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     _,
                     _,
                     _) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_cut = cls._calculate_cut(left_values,
                                                  right_values,
                                                  tree_node.contingency_tables[attrib_index][0],
                                                  tree_node.contingency_tables[attrib_index][1])
                    ret.append((attrib_index,
                                curr_cut,
                                [left_values, right_values],
                                None,
                                timeit.default_timer() - start_time,
                                None,
                                None))
        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!
        best_split_cut = float('-inf')
        best_split_attrib_index = 0
        best_split_left_values = set([])
        best_split_right_values = set([])

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                elif (len(values_seen) > LOG2_LIMIT_EXPONENTIAL_STEPS or
                      (tree_node.number_non_empty_classes
                       * len(values_seen) * 2**len(values_seen)) > LIMIT_EXPONENTIAL_STEPS):
                    print("Attribute {} ({}) is valid but has too many values ({}).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    print("It will be skipped!")
                    continue
                for (left_values,
                     right_values,
                     _,
                     _,
                     _,
                     _) in cls._generate_possible_splits(
                         tree_node.contingency_tables[attrib_index][1],
                         values_seen,
                         tree_node.contingency_tables[attrib_index][0],
                         tree_node.dataset.num_classes):
                    curr_cut = cls._calculate_cut(left_values,
                                                  right_values,
                                                  tree_node.contingency_tables[attrib_index][0],
                                                  tree_node.contingency_tables[attrib_index][1])
                    if curr_cut > best_split_cut:
                        best_split_cut = curr_cut
                        best_split_attrib_index = attrib_index
                        best_split_left_values = left_values
                        best_split_right_values = right_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_split_attrib_index, splits_values, best_split_cut, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_possible_splits(values_num_samples, values_seen, contingency_table, num_classes):
        # MODIFIED!
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
    def _calculate_cut(left_values, right_values, contingency_table, values_num_samples):
        # MODIFIED!
        #TESTED!
        cut_val = 0.0
        num_classes = contingency_table.shape[1]
        num_values = sum(num_samples > 0 for num_samples in values_num_samples)
        for value_left in left_values:
            if values_num_samples[value_left] == 0:
                continue
            for value_right in right_values:
                if values_num_samples[value_right] == 0:
                    continue

                num_samples_both_values = (values_num_samples[value_left]
                                           + values_num_samples[value_right])
                curr_chi_square_value = 0.0
                curr_values_num_classes = 0
                for class_index in range(num_classes):
                    num_samples_both_values_this_class = (
                        contingency_table[value_left][class_index]
                        + contingency_table[value_right][class_index])
                    if num_samples_both_values_this_class == 0:
                        continue
                    curr_values_num_classes += 1
                    expected_value_left_class = (
                        values_num_samples[value_left] * num_samples_both_values_this_class
                        / num_samples_both_values)
                    expected_value_right_class = (
                        values_num_samples[value_right] * num_samples_both_values_this_class
                        / num_samples_both_values)
                    diff_value_left = (contingency_table[value_left][class_index]
                                       - expected_value_left_class)
                    diff_value_right = (contingency_table[value_right][class_index]
                                        - expected_value_right_class)
                    curr_chi_square_value += (
                        diff_value_left * (diff_value_left / expected_value_left_class)
                        + diff_value_right * (diff_value_right / expected_value_right_class))
                    if curr_chi_square_value < 0.0:
                        print('='*90)
                        print('VALOR DE PESO: {} < 0'.format(cut_val))
                        print('='*90)

                if curr_values_num_classes == 1:
                    curr_edge_value = 0.0
                else:
                    curr_edge_value = (
                        chi2.cdf(x=curr_chi_square_value, df=curr_values_num_classes - 1)
                        * (num_samples_both_values / (num_values - 1)))
                cut_val += curr_edge_value

        return cut_val


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
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        #First we pre-calculate the original class frequency and information
        original_information = cls._calculate_information(tree_node.class_index_num_samples,
                                                          len(tree_node.valid_samples_indices))
        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                curr_gain_ratio = cls._calculate_gain_ratio(
                    len(tree_node.valid_samples_indices),
                    tree_node.contingency_tables[attrib_index][0],
                    tree_node.contingency_tables[attrib_index][1],
                    original_information)
                ret.append((attrib_index,
                            curr_gain_ratio,
                            [set([value]) for value in values_seen],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        #TESTED!

        #First we pre-calculate the original class frequency and information
        original_information = cls._calculate_information(tree_node.class_index_num_samples,
                                                          len(tree_node.valid_samples_indices))
        # Since gain_ratio is always non-negative, the starting value below will be replaced in the
        # for loop.
        best_split_gain_ratio = float('-inf')
        best_split_attrib_index = 0
        best_splits_values = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(tree_node.contingency_tables[attrib_index][1])
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue
                curr_gain_ratio = cls._calculate_gain_ratio(
                    len(tree_node.valid_samples_indices),
                    tree_node.contingency_tables[attrib_index][0],
                    tree_node.contingency_tables[attrib_index][1],
                    original_information)
                if curr_gain_ratio > best_split_gain_ratio:
                    best_split_gain_ratio = curr_gain_ratio
                    best_split_attrib_index = attrib_index
                    best_splits_values = [set([value]) for value in values_seen]
        return (best_split_attrib_index, best_splits_values, best_split_gain_ratio, None)

    @staticmethod
    def _get_values_seen(values_num_samples):
        # MODIFIED!
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_gain_ratio(cls, num_valid_samples, contingency_table, values_num_samples,
                              original_information):
        #TESTED!
        information_gain = original_information # Initial information Gain
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            curr_split_information = cls._calculate_information(contingency_table[value],
                                                                value_num_samples)
            information_gain -= (value_num_samples/num_valid_samples) * curr_split_information
        # Gain Ratio
        potential_partition_information = cls._calculate_potential_information(values_num_samples,
                                                                               num_valid_samples)
        gain_ratio = information_gain / potential_partition_information
        return gain_ratio

    @staticmethod
    def _calculate_information(value_class_num_samples, value_num_samples):
        #TESTED!
        information = 0.0
        for curr_class_num_samples in value_class_num_samples:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / value_num_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _calculate_potential_information(values_num_samples, num_valid_samples):
        #TESTED!
        partition_potential_information = 0.0
        for value_num_samples in values_num_samples:
            if value_num_samples != 0:
                curr_ratio = value_num_samples / num_valid_samples
                partition_potential_information -= curr_ratio * math.log2(curr_ratio)
        return partition_potential_information



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                         MAX CUT                                           ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCut(Criterion):
    name = 'Max Cut'


    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret_p_value = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        ret_naive = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 curr_values_histogram,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                ret_naive.append((attrib_index,
                                  curr_gain,
                                  [left_int_values, right_int_values],
                                  None,
                                  timeit.default_timer() - start_time,
                                  None,
                                  None))

                curr_split_histogram = cls._calculate_split_histogram(
                    len(tree_node.valid_samples_indices),
                    curr_values_histogram)
                cover_index = cls._calculate_cover_index(tree_node.dataset.num_classes,
                                                         len(tree_node.valid_samples_indices),
                                                         tree_node.number_samples_in_rarest_class)
                (perfect_gain,
                 num_random_variables) = cls._calculate_perfect_gains(
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                curr_p_value = cls._get_split_p_value(len(tree_node.valid_samples_indices),
                                                      curr_gain,
                                                      curr_split_histogram,
                                                      num_random_variables,
                                                      perfect_gain,
                                                      cover_index)
                ret_p_value.append((attrib_index,
                                    curr_gain,
                                    [left_int_values, right_int_values],
                                    curr_p_value,
                                    timeit.default_timer() - start_time,
                                    None,
                                    None))

        preference_rank_p_value_full = sorted(ret_p_value, key=lambda x: (x[3], -x[1]))
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank_p_value = []
        for pref_elem in preference_rank_p_value_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank_p_value.append(pref_elem)
        ret_with_preference_p_value_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank_p_value):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_p_value_full[attrib_index] = tuple(new_elem)
        ret_with_preference_p_value = [elem
                                       for elem in ret_with_preference_p_value_full if elem != 0]

        preference_rank_naive_full = sorted(ret_naive, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank_naive = []
        for pref_elem in preference_rank_naive_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank_naive.append(pref_elem)
        ret_with_preference_naive_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank_naive):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_naive_full[attrib_index] = tuple(new_elem)
        ret_with_preference_naive = [elem
                                     for elem in ret_with_preference_naive_full if elem != 0]

        return (ret_with_preference_p_value, ret_with_preference_naive)

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_p_value = float('inf')
        best_gain = 0.0
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 curr_values_histogram,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                curr_split_histogram = cls._calculate_split_histogram(
                    len(tree_node.valid_samples_indices),
                    curr_values_histogram)
                cover_index = cls._calculate_cover_index(tree_node.dataset.num_classes,
                                                         len(tree_node.valid_samples_indices),
                                                         tree_node.number_samples_in_rarest_class)
                (perfect_gain,
                 num_random_variables) = cls._calculate_perfect_gains(
                     len(tree_node.valid_samples_indices),
                     tree_node.class_index_num_samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                curr_p_value = cls._get_split_p_value(len(tree_node.valid_samples_indices),
                                                      curr_gain,
                                                      curr_split_histogram,
                                                      num_random_variables,
                                                      perfect_gain,
                                                      cover_index)
                if (curr_p_value < best_p_value
                        or (curr_p_value == best_p_value and curr_gain > best_gain)):
                    best_attrib_index = attrib_index
                    best_p_value = curr_p_value
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, best_p_value)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(num_classes):
                        num_elems_value_j_diff_class = (
                            values_histogram[value_index_j]
                            - values_histogram_with_classes[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            values_histogram_with_classes[value_index_i, class_index]
                            * num_elems_value_j_diff_class)
            return weights


        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        left_values, right_values = cls._generate_random_partition(frac_split_cholesky,
                                                                   new_to_orig_value_int)
        gain = cls._calculate_split_gain(num_classes,
                                         len(valid_samples_indices),
                                         class_index_num_samples,
                                         sample_class,
                                         samples,
                                         attrib_index,
                                         right_values,
                                         diff_keys,
                                         diff_values)
        return gain, values_histogram, left_values, right_values

    @staticmethod
    def _calculate_split_histogram(num_samples, values_histogram):
        #TESTED!
        # Calculates the number of ways to get a split with a given number of values on the
        # left/right side.
        split_histogram = [0] * (num_samples + 1)
        split_histogram[0] = 1
        for num_samples_with_this_value in values_histogram:
            if num_samples_with_this_value > 0:
                for index in range(num_samples, num_samples_with_this_value - 1, -1):
                    split_histogram[index] += split_histogram[index - num_samples_with_this_value]
        return split_histogram

    @staticmethod
    def _calculate_split_gain(num_classes, num_samples, class_index_num_samples, sample_class,
                              samples, attrib_index, right_values, diff_keys, diff_values):
        #TESTED!
        def _init_num_samples_right_split_and_tcv(num_classes, sample_class, samples, attrib_index,
                                                  right_values, diff_keys):
            #TESTED!
            tcv = np.zeros((num_classes, 2), dtype=np.int64)
            # first column = left/false in values_split
            num_samples_right_split = 0
            # tcv[class_index][0] is for samples on the left side of split and tcv[class_index][1]
            # is for samples on the right side.
            for int_key in diff_keys:
                curr_sample_class = sample_class[int_key]
                sample_int_value = samples[int_key][attrib_index]
                if sample_int_value in right_values:
                    num_samples_right_split += 1
                    tcv[curr_sample_class][1] += 1
                else:
                    tcv[curr_sample_class][0] += 1
            return num_samples_right_split, tcv


        # Initialize auxiliary variables
        gain = 0.0
        tc = class_index_num_samples[:] # this slice makes a copy of class_index_num_samples
        num_samples_right_split, tcv = _init_num_samples_right_split_and_tcv(num_classes,
                                                                             sample_class,
                                                                             samples,
                                                                             attrib_index,
                                                                             right_values,
                                                                             diff_keys)
        # Calculate gain and update auxiliary variables

        # Samples we haven't dealt with yet, including the current one. Will subtract 1 at every
        # loop, including first.
        num_remaining_samples = num_samples + 1
        for int_key, sample_diff in zip(diff_keys, diff_values):
            curr_sample_class = sample_class[int_key]
            sample_atrib_int_value = samples[int_key][attrib_index]

            num_remaining_samples -= 1
            num_elems_in_compl_tc = num_remaining_samples - tc[curr_sample_class]

            # Let's calculate the number of samples in same split side (not yet seen in loop) with
            # different class.
            if sample_atrib_int_value in right_values:
                num_elems_compl_tc_same_split = num_samples_right_split - tcv[curr_sample_class][1]
            else:
                num_samples_left_split = num_remaining_samples - num_samples_right_split
                num_elems_compl_tc_same_split = num_samples_left_split - tcv[curr_sample_class][0]

            gain += sample_diff * (num_elems_in_compl_tc - num_elems_compl_tc_same_split)

            # Time to update the auxiliary variables. We decrement tc and tcv so they only have
            # information concerning samples not yet seen in this for loop.
            tc[curr_sample_class] -= 1
            if sample_atrib_int_value in right_values:
                tcv[curr_sample_class][1] -= 1
                num_samples_right_split -= 1
            else:
                tcv[curr_sample_class][0] -= 1
        return gain

    @staticmethod
    def _calculate_perfect_gains(num_samples, class_index_num_samples, sample_class, diff_keys,
                                 diff_values):
        #TESTED!
        # In the common case where the classification errors' costs are 1.0 or 0.0, we have
        # number_random_variables == perfect_gain.
        tc = class_index_num_samples[:]
        perfect_gain = 0.0
        number_random_variables = 0
        for (sample_index,
             (int_key,
              sample_diff)) in enumerate(zip(diff_keys, diff_values)):
            curr_sample_class = sample_class[int_key]

            num_remaining_samples = num_samples - sample_index
            num_elems_in_complement_tc = num_remaining_samples - tc[curr_sample_class]

            perfect_gain += sample_diff * num_elems_in_complement_tc
            number_random_variables += num_elems_in_complement_tc

            tc[curr_sample_class] -= 1
        return perfect_gain, number_random_variables

    @staticmethod
    def _calculate_cover_index(num_classes, num_samples, num_samples_in_rarest_class):
        #TESTED
        if num_classes == 2:
            # Bipartite graph
            return num_samples - num_samples_in_rarest_class
        else:
            return num_samples - num_samples_in_rarest_class + 1

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set([])
        right_values = set([])
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
            else:
                right_values.add(new_to_orig_value_int[new_value])

        return left_values, right_values

    @classmethod
    def _get_split_p_value(cls, num_samples, attrib_gain, split_histogram, num_random_variables,
                           perfect_gain, cover_index):
        #TESTED!
        # We calculate the p-value of each split distribution of samples and sum them weighted by
        # the number of possible ways of getting each one. In other words, use a union bound.
        p_value_sum = 0.0
        for number_left_samples, number_compatible_splits in  enumerate(split_histogram):
            if number_left_samples == 0 or number_left_samples == num_samples:
                continue
            if number_compatible_splits > 0:
                janson_prob = cls._get_janson_prob(number_left_samples,
                                                   num_samples,
                                                   num_random_variables,
                                                   attrib_gain,
                                                   perfect_gain,
                                                   cover_index)
                p_value_sum += number_compatible_splits * janson_prob
        # We have each split possibility counted twice. To see that, just exchange
        # the split's right side with its left side.
        p_value_sum *= 0.5
        return p_value_sum

    @staticmethod
    def _get_janson_prob(num_left_samples, num_samples, num_random_variables, attrib_gain,
                         perfect_gain, cover_index):
        #TESTED!
        def phi_janson(x):
            return ((1.0 + x) * math.log(1.0 + x)) - x

        # From Janson's Corollary 2.4.1

        # The probability of a sample falling in a certain side of the split is given by a
        # binomial distribution with prob (# samples in this side / # total samples). Thus:
        prob_separate = (2.0 * num_left_samples * (num_samples - num_left_samples) /
                         (num_samples**2))

        expected_gain = perfect_gain * prob_separate
        t = attrib_gain - expected_gain
        phi = phi_janson((4.0 * t) / (5.0 * num_random_variables * prob_separate))
        prob = 100.0 * math.exp((- num_random_variables * prob_separate * phi) /
                                ((1.0 - prob_separate) * cover_index))
        return prob



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                      MAX CUT NAIVE                                        ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutNaive(Criterion):
    name = 'Max Cut Naive'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(num_classes):
                        num_elems_value_j_diff_class = (
                            values_histogram[value_index_j]
                            - values_histogram_with_classes[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            values_histogram_with_classes[value_index_i, class_index]
                            * num_elems_value_j_diff_class)
            return weights

        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        left_values, right_values = cls._generate_random_partition(frac_split_cholesky,
                                                                   new_to_orig_value_int)
        gain = cls._calculate_split_gain(num_classes,
                                         len(valid_samples_indices),
                                         class_index_num_samples,
                                         sample_class,
                                         samples,
                                         attrib_index,
                                         right_values,
                                         diff_keys,
                                         diff_values)
        return gain, values_histogram, left_values, right_values

    @staticmethod
    def _calculate_split_gain(num_classes, num_samples, class_index_num_samples, sample_class,
                              samples, attrib_index, right_values, diff_keys, diff_values):
        #TESTED!
        def _init_num_samples_right_split_and_tcv(num_classes, sample_class, samples, attrib_index,
                                                  right_values, diff_keys):
            #TESTED!
            tcv = np.zeros((num_classes, 2), dtype=np.int64)
            # first column = left/false in values_split
            num_samples_right_split = 0
            # tcv[class_index][0] is for samples on the left side of split and tcv[class_index][1]
            # is for samples on the right side.
            for int_key in diff_keys:
                curr_sample_class = sample_class[int_key]
                sample_int_value = samples[int_key][attrib_index]
                if sample_int_value in right_values:
                    num_samples_right_split += 1
                    tcv[curr_sample_class][1] += 1
                else:
                    tcv[curr_sample_class][0] += 1
            return num_samples_right_split, tcv


        # Initialize auxiliary variables
        gain = 0.0
        tc = class_index_num_samples[:] # this slice makes a copy of class_index_num_samples
        num_samples_right_split, tcv = _init_num_samples_right_split_and_tcv(num_classes,
                                                                             sample_class,
                                                                             samples,
                                                                             attrib_index,
                                                                             right_values,
                                                                             diff_keys)
        # Calculate gain and update auxiliary variables

        # Samples we haven't dealt with yet, including the current one. Will subtract 1 at every
        # loop, including first.
        num_remaining_samples = num_samples + 1
        for int_key, sample_diff in zip(diff_keys, diff_values):
            curr_sample_class = sample_class[int_key]
            sample_atrib_int_value = samples[int_key][attrib_index]

            num_remaining_samples -= 1
            num_elems_in_compl_tc = num_remaining_samples - tc[curr_sample_class]

            # Let's calculate the number of samples in same split side (not yet seen in loop) with
            # different class.
            if sample_atrib_int_value in right_values:
                num_elems_compl_tc_same_split = num_samples_right_split - tcv[curr_sample_class][1]
            else:
                num_samples_left_split = num_remaining_samples - num_samples_right_split
                num_elems_compl_tc_same_split = num_samples_left_split - tcv[curr_sample_class][0]

            gain += sample_diff * (num_elems_in_compl_tc - num_elems_compl_tc_same_split)

            # Time to update the auxiliary variables. We decrement tc and tcv so they only have
            # information concerning samples not yet seen in this for loop.
            tc[curr_sample_class] -= 1
            if sample_atrib_int_value in right_values:
                tcv[curr_sample_class][1] -= 1
                num_samples_right_split -= 1
            else:
                tcv[curr_sample_class][0] -= 1
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set([])
        right_values = set([])
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
            else:
                right_values.add(new_to_orig_value_int[new_value])

        return left_values, right_values



#################################################################################################
#################################################################################################
###                                                                                           ###
###                            MAX CUT NAIVE WITH LOCAL SEARCH                                ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutNaiveWithLocalSearch(Criterion):
    name = 'Max Cut Naive With Local Search'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(num_classes):
                        num_elems_value_j_diff_class = (
                            values_histogram[value_index_j]
                            - values_histogram_with_classes[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            values_histogram_with_classes[value_index_i, class_index]
                            * num_elems_value_j_diff_class)
            return weights

        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        (left_values,
         right_values,
         new_left_values,
         new_right_values) = cls._generate_random_partition(frac_split_cholesky,
                                                            new_to_orig_value_int)
        gain = cls._calculate_split_gain(num_classes,
                                         len(valid_samples_indices),
                                         class_index_num_samples,
                                         sample_class,
                                         samples,
                                         attrib_index,
                                         right_values,
                                         diff_keys,
                                         diff_values)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)

        return gain, values_histogram, left_values, right_values



    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain(num_classes, num_samples, class_index_num_samples, sample_class,
                              samples, attrib_index, right_values, diff_keys, diff_values):
        #TESTED!
        def _init_num_samples_right_split_and_tcv(num_classes, sample_class, samples, attrib_index,
                                                  right_values, diff_keys):
            #TESTED!
            tcv = np.zeros((num_classes, 2), dtype=np.int64)
            # first column = left/false in values_split
            num_samples_right_split = 0
            # tcv[class_index][0] is for samples on the left side of split and tcv[class_index][1]
            # is for samples on the right side.
            for int_key in diff_keys:
                curr_sample_class = sample_class[int_key]
                sample_int_value = samples[int_key][attrib_index]
                if sample_int_value in right_values:
                    num_samples_right_split += 1
                    tcv[curr_sample_class][1] += 1
                else:
                    tcv[curr_sample_class][0] += 1
            return num_samples_right_split, tcv


        # Initialize auxiliary variables
        gain = 0.0
        tc = class_index_num_samples[:] # this slice makes a copy of class_index_num_samples
        num_samples_right_split, tcv = _init_num_samples_right_split_and_tcv(num_classes,
                                                                             sample_class,
                                                                             samples,
                                                                             attrib_index,
                                                                             right_values,
                                                                             diff_keys)
        # Calculate gain and update auxiliary variables

        # Samples we haven't dealt with yet, including the current one. Will subtract 1 at every
        # loop, including first.
        num_remaining_samples = num_samples + 1
        for int_key, sample_diff in zip(diff_keys, diff_values):
            curr_sample_class = sample_class[int_key]
            sample_atrib_int_value = samples[int_key][attrib_index]

            num_remaining_samples -= 1
            num_elems_in_compl_tc = num_remaining_samples - tc[curr_sample_class]

            # Let's calculate the number of samples in same split side (not yet seen in loop) with
            # different class.
            if sample_atrib_int_value in right_values:
                num_elems_compl_tc_same_split = num_samples_right_split - tcv[curr_sample_class][1]
            else:
                num_samples_left_split = num_remaining_samples - num_samples_right_split
                num_elems_compl_tc_same_split = num_samples_left_split - tcv[curr_sample_class][0]

            gain += sample_diff * (num_elems_in_compl_tc - num_elems_compl_tc_same_split)

            # Time to update the auxiliary variables. We decrement tc and tcv so they only have
            # information concerning samples not yet seen in this for loop.
            tc[curr_sample_class] -= 1
            if sample_atrib_int_value in right_values:
                tcv[curr_sample_class][1] -= 1
                num_samples_right_split -= 1
            else:
                tcv[curr_sample_class][0] -= 1
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set()
        right_values = set()
        new_left_values = set()
        new_right_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
                new_left_values.add(new_value)
            else:
                right_values.add(new_to_orig_value_int[new_value])
                new_right_values.add(new_value)

        return left_values, right_values, new_left_values, new_right_values



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                   FAST MAX CUT NAIVE                                      ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class FastMaxCutNaive(Criterion):
    name = 'Fast Max Cut Naive'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!
        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

            elif is_valid_numeric_attrib:
                start_time = timeit.default_timer()
                (values_seen,
                 values_and_classes) = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                    tree_node.dataset.samples,
                                                                    tree_node.dataset.sample_class,
                                                                    attrib_index)
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                sorted_values_and_classes = sorted(values_and_classes)
                (best_cut_value,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric(
                     sorted_values_and_classes,
                     tree_node.dataset.num_classes)
                ret.append((attrib_index,
                            best_cut_value,
                            [{last_left_value}, {first_right_value}],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values

            elif is_valid_numeric_attrib:
                (values_seen,
                 values_and_classes) = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                    tree_node.dataset.samples,
                                                                    tree_node.dataset.sample_class,
                                                                    attrib_index)
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                sorted_values_and_classes = sorted(values_and_classes)
                (curr_gain,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric(
                     sorted_values_and_classes,
                     tree_node.dataset.num_classes)
                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = {last_left_value}
                    best_split_right_values = {first_right_value}

        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_seen = set()
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
            if sample_value not in values_seen:
                values_seen.add(sample_value)
        return values_seen, values_and_classes

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(num_classes):
                        num_elems_value_j_diff_class = (
                            values_histogram[value_index_j]
                            - values_histogram_with_classes[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            values_histogram_with_classes[value_index_i, class_index]
                            * num_elems_value_j_diff_class)
            return weights

        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)
        values_seen = set(range(attrib_num_valid_values))

        gain, new_left_values, new_right_values = cls._generate_initial_partition(values_seen,
                                                                                  weights)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)
        else:
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values)
        return gain, values_histogram, left_values, right_values

    @classmethod
    def _generate_initial_partition(cls, values_seen, weights):
        set_left_values = set()
        set_right_values = set()
        cut_val = 0.0

        # calculating initial solution for max cut
        for value in values_seen:
            if len(set_left_values) == 0 and len(set_right_values) == 0:
                set_left_values.add(value)
                continue
            sum_with_left = sum(weights[value][left_value] for left_value in set_left_values)
            sum_with_right = sum(weights[value][right_value] for right_value in set_right_values)
            if sum_with_left >= sum_with_right:
                set_right_values.add(value)
                cut_val += sum_with_left
            else:
                set_left_values.add(value)
                cut_val += sum_with_right
        return cut_val, set_left_values, set_right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @classmethod
    def _best_cut_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        last_left_class = sorted_values_and_classes[0][1]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[last_left_class] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_cut_value = float('-inf')
        best_last_left_value = None
        best_first_right_value = None

        curr_cut_value = num_right_samples - class_num_right[last_left_class]

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            first_right_class = sorted_values_and_classes[first_right_index][1]

            if first_right_value != last_left_value and curr_cut_value > best_cut_value:
                best_cut_value = curr_cut_value
                best_last_left_value = last_left_value
                best_first_right_value = first_right_value

            curr_cut_value -= num_left_samples - class_num_left[first_right_class]
            num_left_samples += 1
            num_right_samples -= 1
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
            curr_cut_value += num_right_samples - class_num_right[first_right_class]
            if first_right_value != last_left_value:
                last_left_value = first_right_value

        return (best_cut_value, best_last_left_value, best_first_right_value)



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                  MAX CUT NAIVE RESIDUE                                    ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutNaiveResidue(Criterion):
    name = 'Max Cut Naive Residue'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(num_classes):
                        num_elems_value_j_diff_class = (
                            values_histogram[value_index_j]
                            - values_histogram_with_classes[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            values_histogram_with_classes[value_index_i, class_index]
                            * num_elems_value_j_diff_class)

                        # Let's subtract the average cut for this pair of values with this
                        # distribution.
                        mixed_class_dist = np.add(values_histogram_with_classes[value_index_i],
                                                  values_histogram_with_classes[value_index_j])

                        total_elems_value_pair = (values_histogram[value_index_i]
                                                  + values_histogram[value_index_j])
                        left_frac = values_histogram[value_index_i] / total_elems_value_pair
                        right_frac = values_histogram[value_index_j] / total_elems_value_pair

                        weights[value_index_i, value_index_j] -= (
                            left_frac * mixed_class_dist[class_index] *
                            (values_histogram[value_index_j]
                             - right_frac * mixed_class_dist[class_index]))

                    if weights[value_index_i, value_index_j] < 0.0:
                        weights[value_index_i, value_index_j] = 0.0

            return weights


        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        (left_values,
         right_values,
         new_left_values,
         new_right_values) = cls._generate_random_partition(frac_split_cholesky,
                                                            new_to_orig_value_int)
        gain = cls._calculate_split_gain(new_left_values,
                                         new_right_values,
                                         weights)
        return gain, values_histogram, left_values, right_values

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set()
        right_values = set()
        new_left_values = set()
        new_right_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
                new_left_values.add(new_value)
            else:
                right_values.add(new_to_orig_value_int[new_value])
                new_right_values.add(new_value)

        return left_values, right_values, new_left_values, new_right_values



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                MAX CUT NAIVE CHI SQUARE                                   ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutNaiveChiSquare(Criterion):
    name = 'Max Cut Naive Chi Square'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                if values_histogram[value_index_i] == 0:
                    continue
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j or values_histogram[value_index_j] == 0:
                        continue

                    num_samples_both_values = (values_histogram[value_index_i]
                                               + values_histogram[value_index_j])
                    for class_index in range(num_classes):
                        num_samples_both_values_this_class = (
                            values_histogram_with_classes[value_index_i, class_index]
                            + values_histogram_with_classes[value_index_j, class_index])
                        if num_samples_both_values_this_class == 0:
                            continue
                        expected_value_index_i_class = (
                            values_histogram[value_index_i] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        expected_value_index_j_class = (
                            values_histogram[value_index_j] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        diff_index_i = (
                            values_histogram_with_classes[value_index_i, class_index]
                            - expected_value_index_i_class)
                        diff_index_j = (
                            values_histogram_with_classes[value_index_j, class_index]
                            - expected_value_index_j_class)

                        edge_weight_curr_class = (
                            diff_index_i * (diff_index_i / expected_value_index_i_class)
                            + diff_index_j * (diff_index_j / expected_value_index_j_class))

                        weights[value_index_i, value_index_j] += edge_weight_curr_class

                        if edge_weight_curr_class < 0.0:
                            print('='*90)
                            print('VALOR DE CHI SQUARE DA ARESTA {}{} COM CLASSE {}: {} < 0'.format(
                                value_index_i,
                                value_index_j,
                                class_index,
                                edge_weight_curr_class))
                            print('='*90)
            return weights


        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        (left_values,
         right_values,
         new_left_values,
         new_right_values) = cls._generate_random_partition(frac_split_cholesky,
                                                            new_to_orig_value_int)
        gain = cls._calculate_split_gain(new_left_values,
                                         new_right_values,
                                         weights)
        return gain, values_histogram, left_values, right_values

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set()
        right_values = set()
        new_left_values = set()
        new_right_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
                new_left_values.add(new_value)
            else:
                right_values.add(new_to_orig_value_int[new_value])
                new_right_values.add(new_value)

        return left_values, right_values, new_left_values, new_right_values



#################################################################################################
#################################################################################################
###                                                                                           ###
###                       MAX CUT NAIVE CHI SQUARE WITH LOCAL SEARCH                          ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutNaiveChiSquareWithLocalSearch(Criterion):
    name = 'Max Cut Naive Chi Square With Local Search'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                if values_histogram[value_index_i] == 0:
                    continue
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j or values_histogram[value_index_j] == 0:
                        continue

                    num_samples_both_values = (values_histogram[value_index_i]
                                               + values_histogram[value_index_j])
                    for class_index in range(num_classes):
                        num_samples_both_values_this_class = (
                            values_histogram_with_classes[value_index_i, class_index]
                            + values_histogram_with_classes[value_index_j, class_index])
                        if num_samples_both_values_this_class == 0:
                            continue
                        expected_value_index_i_class = (
                            values_histogram[value_index_i] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        expected_value_index_j_class = (
                            values_histogram[value_index_j] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        diff_index_i = (
                            values_histogram_with_classes[value_index_i, class_index]
                            - expected_value_index_i_class)
                        diff_index_j = (
                            values_histogram_with_classes[value_index_j, class_index]
                            - expected_value_index_j_class)

                        edge_weight_curr_class = (
                            diff_index_i * (diff_index_i / expected_value_index_i_class)
                            + diff_index_j * (diff_index_j / expected_value_index_j_class))

                        weights[value_index_i, value_index_j] += edge_weight_curr_class

                        if edge_weight_curr_class < 0.0:
                            print('='*90)
                            print('VALOR DE CHI SQUARE DA ARESTA {}{} COM CLASSE {}: {} < 0'.format(
                                value_index_i,
                                value_index_j,
                                class_index,
                                edge_weight_curr_class))
                            print('='*90)
            return weights


        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        (left_values,
         right_values,
         new_left_values,
         new_right_values) = cls._generate_random_partition(frac_split_cholesky,
                                                            new_to_orig_value_int)
        gain = cls._calculate_split_gain(new_left_values,
                                         new_right_values,
                                         weights)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)

        return gain, values_histogram, left_values, right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set()
        right_values = set()
        new_left_values = set()
        new_right_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
                new_left_values.add(new_value)
            else:
                right_values.add(new_to_orig_value_int[new_value])
                new_right_values.add(new_value)

        return left_values, right_values, new_left_values, new_right_values



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                 FAST MAX CUT CHI SQUARE                                   ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class FastMaxCutChiSquare(Criterion):
    name = 'Fast Max Cut Chi Square'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!
        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                if values_histogram[value_index_i] == 0:
                    continue
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j or values_histogram[value_index_j] == 0:
                        continue

                    num_samples_both_values = (values_histogram[value_index_i]
                                               + values_histogram[value_index_j])
                    for class_index in range(num_classes):
                        num_samples_both_values_this_class = (
                            values_histogram_with_classes[value_index_i, class_index]
                            + values_histogram_with_classes[value_index_j, class_index])
                        if num_samples_both_values_this_class == 0:
                            continue
                        expected_value_index_i_class = (
                            values_histogram[value_index_i] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        expected_value_index_j_class = (
                            values_histogram[value_index_j] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        diff_index_i = (
                            values_histogram_with_classes[value_index_i, class_index]
                            - expected_value_index_i_class)
                        diff_index_j = (
                            values_histogram_with_classes[value_index_j, class_index]
                            - expected_value_index_j_class)

                        edge_weight_curr_class = (
                            diff_index_i * (diff_index_i / expected_value_index_i_class)
                            + diff_index_j * (diff_index_j / expected_value_index_j_class))

                        weights[value_index_i, value_index_j] += edge_weight_curr_class

                        if edge_weight_curr_class < 0.0:
                            print('='*90)
                            print('VALOR DE CHI SQUARE DA ARESTA {}{} COM CLASSE {}: {} < 0'.format(
                                value_index_i,
                                value_index_j,
                                class_index,
                                edge_weight_curr_class))
                            print('='*90)
            return weights


        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)
        values_seen = set(range(attrib_num_valid_values))

        gain, new_left_values, new_right_values = cls._generate_initial_partition(values_seen,
                                                                                  weights)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)
        else:
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values)
        return gain, values_histogram, left_values, right_values

    @classmethod
    def _generate_initial_partition(cls, values_seen, weights):
        set_left_values = set()
        set_right_values = set()
        cut_val = 0.0

        # calculating initial solution for max cut
        for value in values_seen:
            if len(set_left_values) == 0 and len(set_right_values) == 0:
                set_left_values.add(value)
                continue
            sum_with_left = sum(weights[value][left_value] for left_value in set_left_values)
            sum_with_right = sum(weights[value][right_value] for right_value in set_right_values)
            if sum_with_left >= sum_with_right:
                set_right_values.add(value)
                cut_val += sum_with_left
            else:
                set_left_values.add(value)
                cut_val += sum_with_right
        return cut_val, set_left_values, set_right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain



#################################################################################################
#################################################################################################
###                                                                                           ###
###                           MAX CUT NAIVE CHI SQUARE NORMALIZED                             ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutNaiveChiSquareNormalized(Criterion):
    name = 'Max Cut Naive Chi Square Normalized'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            num_values = sum(num_samples > 0 for num_samples in values_histogram)
            for value_index_i in range(values_histogram.shape[0]):
                if values_histogram[value_index_i] == 0:
                    continue
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j or values_histogram[value_index_j] == 0:
                        continue

                    num_samples_both_values = (values_histogram[value_index_i]
                                               + values_histogram[value_index_j])
                    for class_index in range(num_classes):
                        num_samples_both_values_this_class = (
                            values_histogram_with_classes[value_index_i, class_index]
                            + values_histogram_with_classes[value_index_j, class_index])
                        if num_samples_both_values_this_class == 0:
                            continue
                        expected_value_index_i_class = (
                            values_histogram[value_index_i] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        expected_value_index_j_class = (
                            values_histogram[value_index_j] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        diff_index_i = (
                            values_histogram_with_classes[value_index_i, class_index]
                            - expected_value_index_i_class)
                        diff_index_j = (
                            values_histogram_with_classes[value_index_j, class_index]
                            - expected_value_index_j_class)

                        edge_weight_curr_class = (
                            diff_index_i * (diff_index_i / expected_value_index_i_class)
                            + diff_index_j * (diff_index_j / expected_value_index_j_class))

                        weights[value_index_i, value_index_j] += edge_weight_curr_class

                        if edge_weight_curr_class < 0.0:
                            print('='*90)
                            print('VALOR DE CHI SQUARE DA ARESTA {}{} COM CLASSE {}: {} < 0'.format(
                                value_index_i,
                                value_index_j,
                                class_index,
                                edge_weight_curr_class))
                            print('='*90)
                    if num_values > 2:
                        weights[value_index_i, value_index_j] /= (num_values - 1.)

            return weights


        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        (left_values,
         right_values,
         new_left_values,
         new_right_values) = cls._generate_random_partition(frac_split_cholesky,
                                                            new_to_orig_value_int)
        gain = cls._calculate_split_gain(new_left_values,
                                         new_right_values,
                                         weights)
        return gain, values_histogram, left_values, right_values

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set()
        right_values = set()
        new_left_values = set()
        new_right_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
                new_left_values.add(new_value)
            else:
                right_values.add(new_to_orig_value_int[new_value])
                new_right_values.add(new_value)

        return left_values, right_values, new_left_values, new_right_values



#################################################################################################
#################################################################################################
###                                                                                           ###
###                   MAX CUT NAIVE CHI SQUARE NORMALIZED WITH LOCAL SEARCH                   ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutNaiveChiSquareNormalizedWithLocalSearch(Criterion):
    name = 'Max Cut Naive Chi Square Normalized With Local Search'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            num_values = sum(num_samples > 0 for num_samples in values_histogram)
            for value_index_i in range(values_histogram.shape[0]):
                if values_histogram[value_index_i] == 0:
                    continue
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j or values_histogram[value_index_j] == 0:
                        continue

                    num_samples_both_values = (values_histogram[value_index_i]
                                               + values_histogram[value_index_j])
                    for class_index in range(num_classes):
                        num_samples_both_values_this_class = (
                            values_histogram_with_classes[value_index_i, class_index]
                            + values_histogram_with_classes[value_index_j, class_index])
                        if num_samples_both_values_this_class == 0:
                            continue
                        expected_value_index_i_class = (
                            values_histogram[value_index_i] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        expected_value_index_j_class = (
                            values_histogram[value_index_j] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        diff_index_i = (
                            values_histogram_with_classes[value_index_i, class_index]
                            - expected_value_index_i_class)
                        diff_index_j = (
                            values_histogram_with_classes[value_index_j, class_index]
                            - expected_value_index_j_class)

                        edge_weight_curr_class = (
                            diff_index_i * (diff_index_i / expected_value_index_i_class)
                            + diff_index_j * (diff_index_j / expected_value_index_j_class))

                        weights[value_index_i, value_index_j] += edge_weight_curr_class

                        if edge_weight_curr_class < 0.0:
                            print('='*90)
                            print('VALOR DE CHI SQUARE DA ARESTA {}{} COM CLASSE {}: {} < 0'.format(
                                value_index_i,
                                value_index_j,
                                class_index,
                                edge_weight_curr_class))
                            print('='*90)
                    if num_values > 2:
                        weights[value_index_i, value_index_j] /= (num_values - 1.)

            return weights


        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        (left_values,
         right_values,
         new_left_values,
         new_right_values) = cls._generate_random_partition(frac_split_cholesky,
                                                            new_to_orig_value_int)
        gain = cls._calculate_split_gain(new_left_values,
                                         new_right_values,
                                         weights)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)
        return gain, values_histogram, left_values, right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set()
        right_values = set()
        new_left_values = set()
        new_right_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
                new_left_values.add(new_value)
            else:
                right_values.add(new_to_orig_value_int[new_value])
                new_right_values.add(new_value)

        return left_values, right_values, new_left_values, new_right_values



#################################################################################################
#################################################################################################
###                                                                                           ###
###                             FAST MAX CUT CHI SQUARE NORMALIZED                            ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class FastMaxCutChiSquareNormalized(Criterion):
    name = 'Fast Max Cut Chi Square Normalized'

    @classmethod
    def evaluate_all_attributes_2(cls, tree_node, num_tests, num_fails_allowed):
        # contains (attrib_index, gain, split_values, p_value, time_taken)
        best_split_per_attrib = []

        num_valid_attrib = 0
        smaller_contingency_tables = {}
        criterion_start_time = timeit.default_timer()

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (orig_to_new_value_int,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue
                smaller_contingency_tables[attrib_index] = (orig_to_new_value_int,
                                                            new_to_orig_value_int,
                                                            smaller_contingency_table,
                                                            smaller_values_num_samples)
                num_valid_attrib += 1
                (curr_gain,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                best_split_per_attrib.append((attrib_index,
                                              curr_gain,
                                              [left_int_values, right_int_values],
                                              None,
                                              timeit.default_timer() - start_time))
        criterion_total_time = timeit.default_timer() - criterion_start_time


        ordered_start_time = timeit.default_timer()
        preference_rank_full = sorted(best_split_per_attrib, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []#(1,
        #                     float('-inf'),
        #                     None,
        #                     None,
        #                     None)]
        # bad_attrib_indices = {3, 5, 6, 10, 11, 12, 13, 17, 18, 20, 21, 22, 25, 56, 57, 52, 55, 59,
        #                       60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 476, 478}
        # preference_rank_mailcode_first = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
            # if pref_elem[0] in bad_attrib_indices:
            #     preference_rank_mailcode_first.append(pref_elem)

        # preference_rank_mailcode_first = sorted(preference_rank_mailcode_first,
        #                                         key=lambda x: x[1])
        # for pref_elem in preference_rank:
        #     if pref_elem[0] in bad_attrib_indices:
        #         continue
        #     preference_rank_mailcode_first.append(pref_elem)

        tests_done_ordered = 0
        accepted_attribute_ordered = None
        ordered_accepted_rank = None
        for (rank_index,
             (attrib_index, best_gain, _, _, _)) in enumerate(preference_rank):
            if math.isinf(best_gain):
                continue
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_ordered += num_tests_needed
            else:
                accepted_attribute_ordered = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_ordered)
                ordered_accepted_rank = rank_index + 1
                tests_done_ordered += num_tests
                break
        ordered_total_time = timeit.default_timer() - ordered_start_time


        rev_start_time = timeit.default_timer()
        # Reversed ordered
        rev_preference_rank = reversed(preference_rank)

        tests_done_rev = 0
        accepted_attribute_rev = None
        for (attrib_index, best_gain, _, _, _) in rev_preference_rank:
            if math.isinf(best_gain):
                continue
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_rev += num_tests_needed
            else:
                accepted_attribute_rev = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_rev)
                tests_done_rev += num_tests
                break
        rev_total_time = timeit.default_timer() - rev_start_time


        # Order splits randomly
        random_start_time = timeit.default_timer()
        random_order_rank = preference_rank[:]
        random.shuffle(random_order_rank)

        tests_done_random_order = 0
        accepted_attribute_random = None
        for (attrib_index, best_gain, _, _, _) in random_order_rank:
            if math.isinf(best_gain):
                continue
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_random_order += num_tests_needed
            else:
                accepted_attribute_random = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_random)
                tests_done_random_order += num_tests
                break
        random_total_time = timeit.default_timer() - random_start_time

        if ordered_accepted_rank is None:
            return (tests_done_ordered,
                    accepted_attribute_ordered,
                    tests_done_rev,
                    accepted_attribute_rev,
                    tests_done_random_order,
                    accepted_attribute_random,
                    num_valid_attrib,
                    ordered_accepted_rank,
                    criterion_total_time,
                    ordered_total_time,
                    rev_total_time,
                    random_total_time,
                    preference_rank[0],
                    None)
        else:
            return (tests_done_ordered,
                    accepted_attribute_ordered,
                    tests_done_rev,
                    accepted_attribute_rev,
                    tests_done_random_order,
                    accepted_attribute_random,
                    num_valid_attrib,
                    ordered_accepted_rank,
                    criterion_total_time,
                    ordered_total_time,
                    rev_total_time,
                    random_total_time,
                    preference_rank[0],
                    preference_rank[ordered_accepted_rank - 1])

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!
        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)

        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                start_time = timeit.default_timer()
                (_,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue
                (curr_gain,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

            elif is_valid_numeric_attrib:
                start_time = timeit.default_timer()
                (values_seen,
                 values_and_classes) = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                    tree_node.dataset.samples,
                                                                    tree_node.dataset.sample_class,
                                                                    attrib_index)
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                sorted_values_and_classes = sorted(values_and_classes)
                (best_cut_value,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric_chi_square(
                     sorted_values_and_classes,
                     tree_node.dataset.num_classes,
                     tree_node.class_index_num_samples)
                ret.append((attrib_index,
                            best_cut_value,
                            [{last_left_value}, {first_right_value}],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])

        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(
                  zip(tree_node.valid_nominal_attribute,
                      tree_node.dataset.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                (_,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue

                (curr_gain,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values

            elif is_valid_numeric_attrib:
                (values_seen,
                 values_and_classes) = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                    tree_node.dataset.samples,
                                                                    tree_node.dataset.sample_class,
                                                                    attrib_index)
                if len(values_seen) <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(values_seen)))
                    continue

                sorted_values_and_classes = sorted(values_and_classes)
                (curr_gain,
                 last_left_value,
                 first_right_value) = cls._best_cut_for_numeric_chi_square(
                     sorted_values_and_classes,
                     tree_node.dataset.num_classes,
                     tree_node.class_index_num_samples)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = {last_left_value}
                    best_split_right_values = {first_right_value}


        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_seen = set()
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
            if sample_value not in values_seen:
                values_seen.add(sample_value)
        return values_seen, values_and_classes

    @classmethod
    def _best_cut_for_numeric_chi_square(cls, sorted_values_and_classes, num_classes,
                                         class_index_num_samples):
        last_left_value = sorted_values_and_classes[0][0]
        last_left_class = sorted_values_and_classes[0][1]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1
        num_samples = len(sorted_values_and_classes)

        class_num_left = [0] * num_classes
        class_num_left[last_left_class] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_cut_value = float('-inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            first_right_class = sorted_values_and_classes[first_right_index][1]

            curr_cut_value = 0.0
            for class_index in range(num_classes):
                if class_index_num_samples[class_index] != 0:
                    expected_value_left_class = (
                        num_left_samples * class_index_num_samples[class_index] / num_samples)
                    diff_left = class_num_left[class_index] - expected_value_left_class
                    curr_cut_value += diff_left * (diff_left / expected_value_left_class)

                    expected_value_right_class = (
                        num_right_samples * class_index_num_samples[class_index] / num_samples)
                    diff_right = class_num_right[class_index] - expected_value_right_class
                    curr_cut_value += diff_right * (diff_right / expected_value_right_class)

            if first_right_value != last_left_value and curr_cut_value > best_cut_value:
                best_cut_value = curr_cut_value
                best_last_left_value = last_left_value
                best_first_right_value = first_right_value
                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1

        return (best_cut_value, best_last_left_value, best_first_right_value)


    @staticmethod
    def _get_smaller_contingency_table(contingency_table, values_num_samples):
        seen_values = set(value
                          for value, num_samples in enumerate(values_num_samples)
                          if num_samples > 0)
        num_classes = contingency_table.shape[1]
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        smaller_contingency_table = np.zeros((len(seen_values), num_classes),
                                             dtype=float)
        smaller_values_num_samples = np.zeros((len(seen_values)), dtype=float)
        for orig_value, num_samples in enumerate(values_num_samples):
            if num_samples == 0:
                continue
            new_value = len(new_to_orig_value_int)
            smaller_values_num_samples[new_value] = num_samples
            orig_to_new_value_int[orig_value] = new_value
            new_to_orig_value_int.append(orig_value)
            smaller_values_num_samples[new_value] = num_samples
            for curr_class, num_samples_curr_class in enumerate(contingency_table[orig_value, :]):
                if num_samples_curr_class > 0:
                    smaller_contingency_table[new_value, curr_class] = num_samples_curr_class

        return (orig_to_new_value_int,
                new_to_orig_value_int,
                smaller_contingency_table,
                smaller_values_num_samples)

    @classmethod
    def _generate_best_split(cls, new_to_orig_value_int, smaller_contingency_table,
                             smaller_values_num_samples):

        def _init_values_weights(contingency_table, values_num_samples):
            # TESTED!
            def _get_chi_square_value(contingency_table_row_1, contingency_table_row_2,
                                      num_samples_first_value, num_samples_second_value):
                ret = 0.0
                num_samples_both_values = num_samples_first_value + num_samples_second_value
                num_classes = contingency_table_row_1.shape[0]
                curr_values_num_classes = 0
                for class_index in range(num_classes):
                    num_samples_both_values_this_class = (
                        contingency_table_row_1[class_index]
                        + contingency_table_row_2[class_index])
                    if num_samples_both_values_this_class == 0:
                        continue
                    curr_values_num_classes += 1

                    expected_value_first_class = (
                        num_samples_first_value * num_samples_both_values_this_class
                        / num_samples_both_values)

                    expected_value_second_class = (
                        num_samples_second_value * num_samples_both_values_this_class
                        / num_samples_both_values)

                    diff_first = (
                        contingency_table_row_1[class_index]
                        - expected_value_first_class)
                    diff_second = (
                        contingency_table_row_2[class_index]
                        - expected_value_second_class)

                    chi_sq_curr_class = (
                        diff_first * (diff_first / expected_value_first_class)
                        + diff_second * (diff_second / expected_value_second_class))

                    ret += chi_sq_curr_class

                    if chi_sq_curr_class < 0.0:
                        print('='*90)
                        print('VALOR DE CHI SQUARE DE UMA ARESTA COM CLASSE {}: {} < 0'.format(
                            class_index,
                            chi_sq_curr_class))
                        print('='*90)
                return ret, curr_values_num_classes

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((smaller_values_num_samples.shape[0], values_num_samples.shape[0]),
                               dtype=np.float64)
            num_values = len(smaller_values_num_samples)
            for value_index_i in range(values_num_samples.shape[0]):
                if values_num_samples[value_index_i] == 0:
                    continue
                for value_index_j in range(values_num_samples.shape[0]):
                    if value_index_i >= value_index_j or values_num_samples[value_index_j] == 0:
                        continue
                    (edge_weight,
                     curr_values_num_classes) = _get_chi_square_value(
                         contingency_table[value_index_i, :],
                         contingency_table[value_index_j, :],
                         values_num_samples[value_index_i],
                         values_num_samples[value_index_j])

                    if curr_values_num_classes == 1:
                        weights[value_index_i, value_index_j] = 0.0
                    else:
                        weights[value_index_i, value_index_j] = edge_weight
                        weights[value_index_j, value_index_i] = edge_weight

                    if num_values > 2:
                        weights[value_index_i, value_index_j] /= (num_values - 1.)
                        weights[value_index_j, value_index_i] = (
                            weights[value_index_i, value_index_j])
            return weights


        weights = _init_values_weights(smaller_contingency_table, smaller_values_num_samples)
        values_seen = set(range(len(new_to_orig_value_int)))

        gain, new_left_values, new_right_values = cls._generate_initial_partition(values_seen,
                                                                                  weights)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)
        else:
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values)
        return gain, left_values, right_values

    @classmethod
    def _generate_initial_partition(cls, values_seen, weights):
        set_left_values = set()
        set_right_values = set()
        cut_val = 0.0

        # calculating initial solution for max cut
        for value in values_seen:
            if len(set_left_values) == 0 and len(set_right_values) == 0:
                set_left_values.add(value)
                continue
            sum_with_left = sum(weights[value][left_value] for left_value in set_left_values)
            sum_with_right = sum(weights[value][right_value] for right_value in set_right_values)
            if sum_with_left >= sum_with_right:
                set_right_values.add(value)
                cut_val += sum_with_left
            else:
                set_left_values.add(value)
                cut_val += sum_with_right
        return cut_val, set_left_values, set_right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def get_classes_dist(contingency_table, values_num_samples, num_valid_samples):
        num_classes = contingency_table.shape[1]
        classes_dist = [0] * num_classes
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index, num_samples in enumerate(contingency_table[value, :]):
                if num_samples > 0:
                    classes_dist[class_index] += num_samples
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)
        return classes_dist

    @staticmethod
    def generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
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
    def accept_attribute(cls, real_gain, num_tests, num_valid_samples, num_fails_allowed,
                         new_to_orig_value_int, smaller_contingency_table,
                         smaller_values_num_samples):
        classes_dist = cls.get_classes_dist(smaller_contingency_table,
                                            smaller_values_num_samples,
                                            num_valid_samples)
        num_fails_seen = 0
        for test_number in range(1, num_tests + 1):
            random_contingency_table = cls.generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                smaller_values_num_samples)
            (gain,
             _,
             _) = cls._generate_best_split(new_to_orig_value_int,
                                           random_contingency_table,
                                           smaller_values_num_samples)
            if gain > real_gain:
                num_fails_seen += 1
                if num_fails_seen > num_fails_allowed:
                    return False, test_number
            if num_tests - test_number <= num_fails_allowed - num_fails_seen:
                return True, None
        return True, None



#################################################################################################
#################################################################################################
###                                                                                           ###
###                          FAST MAX CUT CHI SQUARE NORMALIZED P VALUE                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class FastMaxCutChiSquareNormalizedPValue(Criterion):
    name = 'Fast Max Cut Chi Square Normalized P Value'

    @classmethod
    def evaluate_all_attributes_2(cls, tree_node, num_tests, num_fails_allowed):
        # contains (attrib_index, gain, split_values, p_value, time_taken)
        best_split_per_attrib = []

        num_valid_attrib = 0
        smaller_contingency_tables = {}
        criterion_start_time = timeit.default_timer()

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (orig_to_new_value_int,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue
                smaller_contingency_tables[attrib_index] = (orig_to_new_value_int,
                                                            new_to_orig_value_int,
                                                            smaller_contingency_table,
                                                            smaller_values_num_samples)
                num_valid_attrib += 1
                (curr_gain,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                best_split_per_attrib.append((attrib_index,
                                              curr_gain,
                                              [left_int_values, right_int_values],
                                              None,
                                              timeit.default_timer() - start_time))
        criterion_total_time = timeit.default_timer() - criterion_start_time


        ordered_start_time = timeit.default_timer()
        preference_rank_full = sorted(best_split_per_attrib, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []#(1,
        #                     float('-inf'),
        #                     None,
        #                     None,
        #                     None)]
        # bad_attrib_indices = {3, 5, 6, 10, 11, 12, 13, 17, 18, 20, 21, 22, 25, 56, 57, 52, 55, 59,
        #                       60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 476, 478}
        # preference_rank_mailcode_first = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
            # if pref_elem[0] in bad_attrib_indices:
            #     preference_rank_mailcode_first.append(pref_elem)

        # preference_rank_mailcode_first = sorted(preference_rank_mailcode_first,
        #                                         key=lambda x: x[1])
        # for pref_elem in preference_rank:
        #     if pref_elem[0] in bad_attrib_indices:
        #         continue
        #     preference_rank_mailcode_first.append(pref_elem)

        tests_done_ordered = 0
        accepted_attribute_ordered = None
        ordered_accepted_rank = None
        for (rank_index,
             (attrib_index, best_gain, _, _, _)) in enumerate(preference_rank):
            if math.isinf(best_gain):
                continue
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_ordered += num_tests_needed
            else:
                accepted_attribute_ordered = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_ordered)
                ordered_accepted_rank = rank_index + 1
                tests_done_ordered += num_tests
                break
        ordered_total_time = timeit.default_timer() - ordered_start_time


        rev_start_time = timeit.default_timer()
        # Reversed ordered
        rev_preference_rank = reversed(preference_rank)

        tests_done_rev = 0
        accepted_attribute_rev = None
        for (attrib_index, best_gain, _, _, _) in rev_preference_rank:
            if math.isinf(best_gain):
                continue
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_rev += num_tests_needed
            else:
                accepted_attribute_rev = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_rev)
                tests_done_rev += num_tests
                break
        rev_total_time = timeit.default_timer() - rev_start_time


        # Order splits randomly
        random_start_time = timeit.default_timer()
        random_order_rank = preference_rank[:]
        random.shuffle(random_order_rank)

        tests_done_random_order = 0
        accepted_attribute_random = None
        for (attrib_index, best_gain, _, _, _) in random_order_rank:
            if math.isinf(best_gain):
                continue
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_random_order += num_tests_needed
            else:
                accepted_attribute_random = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_random)
                tests_done_random_order += num_tests
                break
        random_total_time = timeit.default_timer() - random_start_time

        if ordered_accepted_rank is None:
            return (tests_done_ordered,
                    accepted_attribute_ordered,
                    tests_done_rev,
                    accepted_attribute_rev,
                    tests_done_random_order,
                    accepted_attribute_random,
                    num_valid_attrib,
                    ordered_accepted_rank,
                    criterion_total_time,
                    ordered_total_time,
                    rev_total_time,
                    random_total_time,
                    preference_rank[0],
                    None)
        else:
            return (tests_done_ordered,
                    accepted_attribute_ordered,
                    tests_done_rev,
                    accepted_attribute_rev,
                    tests_done_random_order,
                    accepted_attribute_random,
                    num_valid_attrib,
                    ordered_accepted_rank,
                    criterion_total_time,
                    ordered_total_time,
                    rev_total_time,
                    random_total_time,
                    preference_rank[0],
                    preference_rank[ordered_accepted_rank - 1])

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!
        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (_,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue
                (curr_gain,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (_,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue

                (curr_gain,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_smaller_contingency_table(contingency_table, values_num_samples):
        seen_values = set(value
                          for value, num_samples in enumerate(values_num_samples)
                          if num_samples > 0)
        num_classes = contingency_table.shape[1]
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        smaller_contingency_table = np.zeros((len(seen_values), num_classes),
                                             dtype=float)
        smaller_values_num_samples = np.zeros((len(seen_values)), dtype=float)
        for orig_value, num_samples in enumerate(values_num_samples):
            if num_samples == 0:
                continue
            new_value = len(new_to_orig_value_int)
            smaller_values_num_samples[new_value] = num_samples
            orig_to_new_value_int[orig_value] = new_value
            new_to_orig_value_int.append(orig_value)
            smaller_values_num_samples[new_value] = num_samples
            for curr_class, num_samples_curr_class in enumerate(contingency_table[orig_value, :]):
                if num_samples_curr_class > 0:
                    smaller_contingency_table[new_value, curr_class] = num_samples_curr_class

        return (orig_to_new_value_int,
                new_to_orig_value_int,
                smaller_contingency_table,
                smaller_values_num_samples)

    @classmethod
    def _generate_best_split(cls, new_to_orig_value_int, smaller_contingency_table,
                             smaller_values_num_samples):

        def _init_values_weights(contingency_table, values_num_samples):
            # TESTED!
            def _get_chi_square_value(contingency_table_row_1, contingency_table_row_2,
                                      num_samples_first_value, num_samples_second_value):
                ret = 0.0
                num_samples_both_values = num_samples_first_value + num_samples_second_value
                num_classes = contingency_table_row_1.shape[0]
                curr_values_num_classes = 0
                for class_index in range(num_classes):
                    num_samples_both_values_this_class = (
                        contingency_table_row_1[class_index]
                        + contingency_table_row_2[class_index])
                    if num_samples_both_values_this_class == 0:
                        continue
                    curr_values_num_classes += 1

                    expected_value_first_class = (
                        num_samples_first_value * num_samples_both_values_this_class
                        / num_samples_both_values)

                    expected_value_second_class = (
                        num_samples_second_value * num_samples_both_values_this_class
                        / num_samples_both_values)

                    diff_first = (
                        contingency_table_row_1[class_index]
                        - expected_value_first_class)
                    diff_second = (
                        contingency_table_row_2[class_index]
                        - expected_value_second_class)

                    chi_sq_curr_class = (
                        diff_first * (diff_first / expected_value_first_class)
                        + diff_second * (diff_second / expected_value_second_class))

                    ret += chi_sq_curr_class

                    if chi_sq_curr_class < 0.0:
                        print('='*90)
                        print('VALOR DE CHI SQUARE DE UMA ARESTA COM CLASSE {}: {} < 0'.format(
                            class_index,
                            chi_sq_curr_class))
                        print('='*90)
                return ret, curr_values_num_classes


            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((smaller_values_num_samples.shape[0], values_num_samples.shape[0]),
                               dtype=np.float64)
            num_values = len(smaller_values_num_samples)
            for value_index_i in range(values_num_samples.shape[0]):
                if values_num_samples[value_index_i] == 0:
                    continue
                for value_index_j in range(values_num_samples.shape[0]):
                    if value_index_i >= value_index_j or values_num_samples[value_index_j] == 0:
                        continue
                    (edge_weight,
                     curr_values_num_classes) = _get_chi_square_value(
                         contingency_table[value_index_i, :],
                         contingency_table[value_index_j, :],
                         values_num_samples[value_index_i],
                         values_num_samples[value_index_j])
                    num_samples_both_values = (values_num_samples[value_index_i]
                                               + values_num_samples[value_index_j])

                    if curr_values_num_classes == 1:
                        weights[value_index_i, value_index_j] = 0.0
                    else:
                        weights[value_index_i, value_index_j] = edge_weight * chi2.cdf(
                            x=edge_weight,
                            df=curr_values_num_classes - 1)
                        weights[value_index_j, value_index_i] = (
                            weights[value_index_i, value_index_j])

                    if num_values > 2:
                        weights[value_index_i, value_index_j] /= (num_values - 1.)
                        weights[value_index_j, value_index_i] = (
                            weights[value_index_i, value_index_j])
            return weights


        weights = _init_values_weights(smaller_contingency_table, smaller_values_num_samples)
        values_seen = set(range(len(new_to_orig_value_int)))

        gain, new_left_values, new_right_values = cls._generate_initial_partition(values_seen,
                                                                                  weights)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)
        else:
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values)
        return gain, left_values, right_values

    @classmethod
    def _generate_initial_partition(cls, values_seen, weights):
        set_left_values = set()
        set_right_values = set()
        cut_val = 0.0

        # calculating initial solution for max cut
        for value in values_seen:
            if len(set_left_values) == 0 and len(set_right_values) == 0:
                set_left_values.add(value)
                continue
            sum_with_left = sum(weights[value][left_value] for left_value in set_left_values)
            sum_with_right = sum(weights[value][right_value] for right_value in set_right_values)
            if sum_with_left >= sum_with_right:
                set_right_values.add(value)
                cut_val += sum_with_left
            else:
                set_left_values.add(value)
                cut_val += sum_with_right
        return cut_val, set_left_values, set_right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def get_classes_dist(contingency_table, values_num_samples, num_valid_samples):
        num_classes = contingency_table.shape[1]
        classes_dist = [0] * num_classes
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index, num_samples in enumerate(contingency_table[value, :]):
                if num_samples > 0:
                    classes_dist[class_index] += num_samples
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)
        return classes_dist

    @staticmethod
    def generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
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
    def accept_attribute(cls, real_gain, num_tests, num_valid_samples, num_fails_allowed,
                         new_to_orig_value_int, smaller_contingency_table,
                         smaller_values_num_samples):
        classes_dist = cls.get_classes_dist(smaller_contingency_table,
                                            smaller_values_num_samples,
                                            num_valid_samples)
        num_fails_seen = 0
        for test_number in range(1, num_tests + 1):
            random_contingency_table = cls.generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                smaller_values_num_samples)
            (gain,
             _,
             _) = cls._generate_best_split(new_to_orig_value_int,
                                           random_contingency_table,
                                           smaller_values_num_samples)
            if gain > real_gain:
                num_fails_seen += 1
                if num_fails_seen > num_fails_allowed:
                    return False, test_number
            if num_tests - test_number <= num_fails_allowed - num_fails_seen:
                return True, None
        return True, None



#################################################################################################
#################################################################################################
###                                                                                           ###
###                    FAST MAX CUT CHI SQUARE NORMALIZED P VALUE M C                         ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class FastMaxCutChiSquareNormalizedPValueMC(Criterion):
    name = 'Fast Max Cut Chi Square Normalized P Value M C'


    @classmethod
    def evaluate_all_attributes_2(cls, tree_node, num_tests, num_fails_allowed):
        # contains (attrib_index, gain, split_values, p_value, time_taken)
        best_split_per_attrib = []

        num_valid_attrib = 0
        smaller_contingency_tables = {}

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (orig_to_new_value_int,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue
                smaller_contingency_tables[attrib_index] = (orig_to_new_value_int,
                                                            new_to_orig_value_int,
                                                            smaller_contingency_table,
                                                            smaller_values_num_samples)
                num_valid_attrib += 1

                (curr_gain,
                 left_int_values,
                 right_int_values,
                 _) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                best_split_per_attrib.append((attrib_index,
                                              curr_gain,
                                              [left_int_values, right_int_values],
                                              None,
                                              timeit.default_timer() - start_time))

        # Order splits by gini value
        preference_rank_full = sorted(best_split_per_attrib, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []#(1,
        #                     float('-inf'),
        #                     None,
        #                     None,
        #                     None)]
        # bad_attrib_indices = {3, 5, 6, 10, 11, 12, 13, 17, 18, 20, 21, 22, 25, 56, 57, 52, 55, 59,
        #                       60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 476, 478}
        # preference_rank_mailcode_first = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
            # if pref_elem[0] in bad_attrib_indices:
            #     preference_rank_mailcode_first.append(pref_elem)

        # preference_rank_mailcode_first = sorted(preference_rank_mailcode_first,
        #                                         key=lambda x: x[1])
        # for pref_elem in preference_rank:
        #     if pref_elem[0] in bad_attrib_indices:
        #         continue
        #     preference_rank_mailcode_first.append(pref_elem)

        tests_done_ordered = 0
        accepted_attribute_ordered = None
        for (attrib_index, best_gain, _, _, _) in preference_rank:
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_ordered += num_tests_needed
            else:
                accepted_attribute_ordered = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_ordered)
                tests_done_ordered += num_tests
                break

        # Reversed ordered
        rev_preference_rank = reversed(preference_rank)

        tests_done_rev = 0
        accepted_attribute_rev = None
        for (attrib_index, best_gain, _, _, _) in rev_preference_rank:
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_rev += num_tests_needed
            else:
                accepted_attribute_rev = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_rev)
                tests_done_rev += num_tests
                break

        # Order splits randomly
        random_order_rank = preference_rank[:]
        random.shuffle(random_order_rank)

        tests_done_random_order = 0
        accepted_attribute_random = None
        for (attrib_index, best_gain, _, _, _) in random_order_rank:
            (_,
             new_to_orig_value_int,
             smaller_contingency_table,
             smaller_values_num_samples) = smaller_contingency_tables[attrib_index]
            (should_accept,
             num_tests_needed) = cls.accept_attribute(
                 best_gain,
                 num_tests,
                 len(tree_node.valid_samples_indices),
                 num_fails_allowed,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples)
            if not should_accept:
                tests_done_random_order += num_tests_needed
            else:
                accepted_attribute_random = tree_node.dataset.attrib_names[attrib_index]
                print('Accepted attribute:', accepted_attribute_random)
                tests_done_random_order += num_tests
                break

        return (tests_done_ordered,
                accepted_attribute_ordered,
                tests_done_rev,
                accepted_attribute_rev,
                tests_done_random_order,
                accepted_attribute_random,
                num_valid_attrib)

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!
        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (_,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue
                (curr_gain,
                 left_int_values,
                 right_int_values,
                 num_monte_carlo_done) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None,
                            num_monte_carlo_done))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (_,
                 new_to_orig_value_int,
                 smaller_contingency_table,
                 smaller_values_num_samples) = cls._get_smaller_contingency_table(
                     tree_node.contingency_tables[attrib_index][0],
                     tree_node.contingency_tables[attrib_index][1])
                if len(new_to_orig_value_int) <= 1:
                    print("Attribute {} ({}) is valid but only has {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        len(new_to_orig_value_int)))
                    continue

                (curr_gain,
                 left_int_values,
                 right_int_values,
                 _) = cls._generate_best_split(
                     new_to_orig_value_int,
                     smaller_contingency_table,
                     smaller_values_num_samples)
                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_smaller_contingency_table(contingency_table, values_num_samples):
        seen_values = set(value
                          for value, num_samples in enumerate(values_num_samples)
                          if num_samples > 0)
        num_classes = contingency_table.shape[1]
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        smaller_contingency_table = np.zeros((len(seen_values), num_classes),
                                             dtype=float)
        smaller_values_num_samples = np.zeros((len(seen_values)), dtype=float)
        for orig_value, num_samples in enumerate(values_num_samples):
            if num_samples == 0:
                continue
            new_value = len(new_to_orig_value_int)
            smaller_values_num_samples[new_value] = num_samples
            orig_to_new_value_int[orig_value] = new_value
            new_to_orig_value_int.append(orig_value)
            smaller_values_num_samples[new_value] = num_samples
            for curr_class, num_samples_curr_class in enumerate(contingency_table[orig_value, :]):
                if num_samples_curr_class > 0:
                    smaller_contingency_table[new_value, curr_class] = num_samples_curr_class

        return (orig_to_new_value_int,
                new_to_orig_value_int,
                smaller_contingency_table,
                smaller_values_num_samples)

    @classmethod
    def _generate_best_split(cls, new_to_orig_value_int, smaller_contingency_table,
                             smaller_values_num_samples):

        def _init_values_weights(contingency_table, values_num_samples):
            # TESTED!
            def _monte_carlo_p_value(num_samples_first_value, num_samples_second_value,
                                     both_values_class_distribution, num_classes,
                                     orig_chi_square_value):
                num_better = 0
                for _ in range(NUM_TESTS_CHI_SQUARE_MONTE_CARLO):
                    num_samples_both_values = num_samples_first_value + num_samples_second_value
                    random_classes = np.random.choice(len(both_values_class_distribution),
                                                      num_samples_both_values,
                                                      replace=True,
                                                      p=both_values_class_distribution)
                    contingency_table_first_val = np.zeros((num_classes), dtype=float)
                    contingency_table_second_val = np.zeros((num_classes), dtype=float)
                    for random_class in random_classes[:num_samples_first_value]:
                        contingency_table_first_val[random_class] += 1
                    for random_class in random_classes[num_samples_first_value:]:
                        contingency_table_second_val[random_class] += 1

                    (chi_square_value,
                     _,
                     _,
                     _) = _get_chi_square_value(
                         contingency_table_first_val,
                         contingency_table_second_val,
                         num_samples_first_value,
                         num_samples_second_value)
                    if chi_square_value > orig_chi_square_value:
                        num_better += 1

                return num_better / NUM_TESTS_CHI_SQUARE_MONTE_CARLO

            def _get_chi_square_value(contingency_table_row_1, contingency_table_row_2,
                                      num_samples_first_value, num_samples_second_value):
                ret = 0.0
                num_samples_both_values = num_samples_first_value + num_samples_second_value
                num_classes = contingency_table_row_1.shape[0]
                curr_values_num_classes = 0
                num_below_5 = 0
                num_below_10 = 0
                for class_index in range(num_classes):
                    num_samples_both_values_this_class = (
                        contingency_table_row_1[class_index]
                        + contingency_table_row_2[class_index])
                    if num_samples_both_values_this_class == 0:
                        continue
                    curr_values_num_classes += 1

                    expected_value_first_class = (
                        num_samples_first_value * num_samples_both_values_this_class
                        / num_samples_both_values)
                    if expected_value_first_class < 5.0:
                        num_below_5 += 1
                        num_below_10 += 1
                    elif expected_value_first_class < 10.0:
                        num_below_10 += 1

                    expected_value_second_class = (
                        num_samples_second_value * num_samples_both_values_this_class
                        / num_samples_both_values)
                    if expected_value_second_class < 5.0:
                        num_below_5 += 1
                        num_below_10 += 1
                    elif expected_value_second_class < 10.0:
                        num_below_10 += 1

                    diff_first = (
                        contingency_table_row_1[class_index]
                        - expected_value_first_class)
                    diff_second = (
                        contingency_table_row_2[class_index]
                        - expected_value_second_class)

                    chi_sq_curr_class = (
                        diff_first * (diff_first / expected_value_first_class)
                        + diff_second * (diff_second / expected_value_second_class))

                    ret += chi_sq_curr_class

                    if chi_sq_curr_class < 0.0:
                        print('='*90)
                        print('VALOR DE CHI SQUARE DE UMA ARESTA COM CLASSE {}: {} < 0'.format(
                            class_index,
                            chi_sq_curr_class))
                        print('='*90)
                return ret, num_below_5, num_below_10, curr_values_num_classes


            def _get_both_values_class_distribution(contingency_table_row_1,
                                                    contingency_table_row_2,
                                                    num_samples_both_values):
                num_classes = contingency_table_row_1.shape[0]
                class_dist = [0] * num_classes
                for class_index, num_samples_curr_class in enumerate(contingency_table_row_1):
                    if num_samples_curr_class > 0:
                        class_dist[class_index] += num_samples_curr_class
                for class_index, num_samples_curr_class in enumerate(contingency_table_row_2):
                    if num_samples_curr_class > 0:
                        class_dist[class_index] += num_samples_curr_class
                for class_index in range(num_classes):
                    class_dist[class_index] /= num_samples_both_values
                return class_dist

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((smaller_values_num_samples.shape[0], values_num_samples.shape[0]),
                               dtype=np.float64)
            num_values = len(smaller_values_num_samples)
            num_monte_carlo_done = 0
            for value_index_i in range(values_num_samples.shape[0]):
                if values_num_samples[value_index_i] == 0:
                    continue
                for value_index_j in range(values_num_samples.shape[0]):
                    if value_index_i >= value_index_j or values_num_samples[value_index_j] == 0:
                        continue
                    (edge_weight,
                     num_below_5,
                     num_below_10,
                     curr_values_num_classes) = _get_chi_square_value(
                         contingency_table[value_index_i, :],
                         contingency_table[value_index_j, :],
                         values_num_samples[value_index_i],
                         values_num_samples[value_index_j])
                    num_samples_both_values = (values_num_samples[value_index_i]
                                               + values_num_samples[value_index_j])

                    if curr_values_num_classes == 1:
                        weights[value_index_i, value_index_j] = 0.0
                    # number of of entries on the contingency table = #values * #classes
                    elif ((curr_values_num_classes == 2
                           and num_below_10 > 0.2 * 2 * curr_values_num_classes)
                          or num_below_5 > 0.2 * 2 * curr_values_num_classes):
                        num_monte_carlo_done += 1
                        both_values_class_distribution = _get_both_values_class_distribution(
                            contingency_table[value_index_i, :],
                            contingency_table[value_index_j, :],
                            num_samples_both_values)
                        p_value_monte_carlo = _monte_carlo_p_value(
                            int(values_num_samples[value_index_i]),
                            int(values_num_samples[value_index_j]),
                            both_values_class_distribution,
                            contingency_table.shape[1],
                            edge_weight)

                        weights[value_index_i, value_index_j] = (
                            edge_weight * (1. - p_value_monte_carlo))
                        weights[value_index_j, value_index_i] = (
                            weights[value_index_i, value_index_j])
                    else:
                        weights[value_index_i, value_index_j] = edge_weight * chi2.cdf(
                            x=edge_weight,
                            df=curr_values_num_classes - 1)
                        weights[value_index_j, value_index_i] = (
                            weights[value_index_i, value_index_j])

                    if num_values > 2:
                        weights[value_index_i, value_index_j] /= (num_values - 1.)
                        weights[value_index_j, value_index_i] = (
                            weights[value_index_i, value_index_j])
            return weights, num_monte_carlo_done


        (weights,
         num_monte_carlo_done) = _init_values_weights(smaller_contingency_table,
                                                      smaller_values_num_samples)
        values_seen = set(range(len(new_to_orig_value_int)))

        gain, new_left_values, new_right_values = cls._generate_initial_partition(values_seen,
                                                                                  weights)
        # Look for a better solution locally
        (gain_switched,
         new_left_values_switched,
         new_right_values_switched) = cls._switch_while_increase(gain,
                                                                 new_left_values,
                                                                 new_right_values,
                                                                 weights)
        if gain_switched > gain:
            gain = gain_switched
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values_switched)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values_switched)
        else:
            left_values = set(new_to_orig_value_int[new_value]
                              for new_value in new_left_values)
            right_values = set(new_to_orig_value_int[new_value]
                               for new_value in new_right_values)
        return gain, left_values, right_values, num_monte_carlo_done

    @classmethod
    def _generate_initial_partition(cls, values_seen, weights):
        set_left_values = set()
        set_right_values = set()
        cut_val = 0.0

        # calculating initial solution for max cut
        for value in values_seen:
            if len(set_left_values) == 0 and len(set_right_values) == 0:
                set_left_values.add(value)
                continue
            sum_with_left = sum(weights[value][left_value] for left_value in set_left_values)
            sum_with_right = sum(weights[value][right_value] for right_value in set_right_values)
            if sum_with_left >= sum_with_right:
                set_right_values.add(value)
                cut_val += sum_with_left
            else:
                set_left_values.add(value)
                cut_val += sum_with_right
        return cut_val, set_left_values, set_right_values

    @classmethod
    def _switch_while_increase(cls, cut_val, set_left_values, set_right_values, weights):
        curr_cut_val = cut_val
        values_seen = set_left_values | set_right_values

        improvement = True
        while improvement:
            improvement = False
            for value in values_seen:
                new_cut_val = cls._calculate_split_gain_for_single_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          value,
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value in set_left_values:
                        set_left_values.remove(value)
                        set_right_values.add(value)
                    else:
                        set_left_values.add(value)
                        set_right_values.remove(value)
                    improvement = True
                    break
            if improvement:
                continue
            for value1, value2 in itertools.combinations(values_seen, 2):
                if ((value1 in set_left_values and value2 in set_left_values) or
                        (value1 in set_right_values and value2 in set_right_values)):
                    continue
                new_cut_val = cls._calculate_split_gain_for_double_switch(curr_cut_val,
                                                                          set_left_values,
                                                                          set_right_values,
                                                                          (value1, value2),
                                                                          weights)
                if new_cut_val - curr_cut_val > 0.000001:
                    curr_cut_val = new_cut_val
                    if value1 in set_left_values:
                        set_left_values.remove(value1)
                        set_right_values.add(value1)
                        set_right_values.remove(value2)
                        set_left_values.add(value2)
                    else:
                        set_left_values.remove(value2)
                        set_right_values.add(value2)
                        set_right_values.remove(value1)
                        set_left_values.add(value1)
                    improvement = True
                    break

        return curr_cut_val, set_left_values, set_right_values

    @staticmethod
    def _calculate_split_gain_for_single_switch(curr_gain, new_left_values, new_right_values,
                                                new_value_to_change_sides, weights):
        new_gain = curr_gain
        if new_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
            for value in new_right_values:
                new_gain -= weights[value][new_value_to_change_sides]
        else:
            for value in new_left_values:
                new_gain -= weights[value][new_value_to_change_sides]
            for value in new_right_values:
                if value == new_value_to_change_sides:
                    continue
                new_gain += weights[value][new_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain_for_double_switch(curr_gain, new_left_values, new_right_values,
                                                new_values_to_change_sides, weights):
        assert len(new_values_to_change_sides) == 2
        new_gain = curr_gain
        first_value_to_change_sides = new_values_to_change_sides[0]
        second_value_to_change_sides = new_values_to_change_sides[1]

        if first_value_to_change_sides in new_left_values:
            for value in new_left_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
        else:
            for value in new_left_values:
                if value == second_value_to_change_sides:
                    continue
                new_gain -= weights[value][first_value_to_change_sides]
                new_gain += weights[value][second_value_to_change_sides]
            for value in new_right_values:
                if value == first_value_to_change_sides:
                    continue
                new_gain += weights[value][first_value_to_change_sides]
                new_gain -= weights[value][second_value_to_change_sides]
        return new_gain

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def get_classes_dist(contingency_table, values_num_samples, num_valid_samples):
        num_classes = contingency_table.shape[1]
        classes_dist = [0] * num_classes
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index, num_samples in enumerate(contingency_table[value, :]):
                if num_samples > 0:
                    classes_dist[class_index] += num_samples
        for class_index in range(num_classes):
            classes_dist[class_index] /= float(num_valid_samples)
        return classes_dist

    @staticmethod
    def generate_random_contingency_table(classes_dist, num_valid_samples, values_num_samples):
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
    def accept_attribute(cls, real_gain, num_tests, num_valid_samples, num_fails_allowed,
                         new_to_orig_value_int, smaller_contingency_table,
                         smaller_values_num_samples):
        classes_dist = cls.get_classes_dist(smaller_contingency_table,
                                            smaller_values_num_samples,
                                            num_valid_samples)
        num_fails_seen = 0
        for test_number in range(1, num_tests + 1):
            random_contingency_table = cls.generate_random_contingency_table(
                classes_dist,
                num_valid_samples,
                smaller_values_num_samples)
            (gain,
             _,
             _,
             _) = cls._generate_best_split(new_to_orig_value_int,
                                           random_contingency_table,
                                           smaller_values_num_samples)
            if gain > real_gain:
                num_fails_seen += 1
                if num_fails_seen > num_fails_allowed:
                    return False, test_number
            if num_tests - test_number <= num_fails_allowed - num_fails_seen:
                return True, None
        return True, None



#################################################################################################
#################################################################################################
###                                                                                           ###
###                           MAX CUT NAIVE CHI SQUARE HEURISTIC                              ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutNaiveChiSquareHeuristic(Criterion):
    name = 'Max Cut Naive Chi Square Heuristic'

    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            None,
                            None))

        preference_rank_full = sorted(ret, key=lambda x: -x[1])
        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)
        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     attrib_index,
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     tree_node.dataset.samples,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)

                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @classmethod
    def _generate_best_split(cls, attrib_index, num_classes, attrib_num_valid_values,
                             orig_to_new_value_int, new_to_orig_value_int, valid_samples_indices,
                             class_index_num_samples, samples, sample_class, diff_keys,
                             diff_values):
        #TESTED!
        def _init_values_histograms(attrib_index, num_classes, attrib_num_valid_values,
                                    valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples[sample_index][attrib_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!

            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            num_values = sum(num_samples > 0 for num_samples in values_histogram)
            for value_index_i in range(values_histogram.shape[0]):
                if values_histogram[value_index_i] == 0:
                    continue
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j or values_histogram[value_index_j] == 0:
                        continue

                    num_samples_both_values = (values_histogram[value_index_i]
                                               + values_histogram[value_index_j])
                    curr_chi_square_value = 0.0
                    curr_values_num_classes = 0
                    for class_index in range(num_classes):
                        num_samples_both_values_this_class = (
                            values_histogram_with_classes[value_index_i, class_index]
                            + values_histogram_with_classes[value_index_j, class_index])
                        if num_samples_both_values_this_class == 0:
                            continue
                        curr_values_num_classes += 1
                        expected_value_index_i_class = (
                            values_histogram[value_index_i] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        expected_value_index_j_class = (
                            values_histogram[value_index_j] * num_samples_both_values_this_class
                            / num_samples_both_values)
                        diff_index_i = (
                            values_histogram_with_classes[value_index_i, class_index]
                            - expected_value_index_i_class)
                        diff_index_j = (
                            values_histogram_with_classes[value_index_j, class_index]
                            - expected_value_index_j_class)
                        curr_chi_square_value += (
                            diff_index_i * (diff_index_i / expected_value_index_i_class)
                            + diff_index_j * (diff_index_j / expected_value_index_j_class))
                        if curr_chi_square_value < 0.0:
                            print('='*90)
                            print('VALOR DE CHI SQUARE DA ARESTA {}{} COM CLASSE {}: {} < 0'.format(
                                value_index_i,
                                value_index_j,
                                class_index,
                                curr_chi_square_value))
                            print('='*90)
                    if curr_values_num_classes == 1:
                        curr_edge_value = 0.0
                    else:
                        curr_edge_value = (
                            chi2.cdf(x=curr_chi_square_value, df=curr_values_num_classes - 1)
                            * (num_samples_both_values / (num_values - 1)))
                    weights[value_index_i, value_index_j] = curr_edge_value

            return weights


        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(attrib_index,
                                                                  num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        (left_values,
         right_values,
         new_left_values,
         new_right_values) = cls._generate_random_partition(frac_split_cholesky,
                                                            new_to_orig_value_int)
        gain = cls._calculate_split_gain(new_left_values,
                                         new_right_values,
                                         weights)
        return gain, values_histogram, left_values, right_values

    @staticmethod
    def _calculate_split_gain(new_left_values, new_right_values, weights):
        gain = 0.0
        for value_left, value_right in itertools.product(new_left_values, new_right_values):
            gain += weights[value_left, value_right]
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set()
        right_values = set()
        new_left_values = set()
        new_right_values = set()
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
                new_left_values.add(new_value)
            else:
                right_values.add(new_to_orig_value_int[new_value])
                new_right_values.add(new_value)

        return left_values, right_values, new_left_values, new_right_values



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                  MAX CUT MONTE CARLO                                      ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutMonteCarlo(Criterion):
    name = 'Max Cut Monte Carlo'


    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken,
                 #           should_accept, num_tests_needed)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        num_valid_attrib = sum(tree_node.valid_nominal_attribute)
        num_tests = int(math.ceil(math.log2(num_valid_attrib))) + 6

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                samples_orig_values = cls._generate_samples_orig_values(
                    attrib_index,
                    tree_node.valid_samples_indices,
                    tree_node.dataset.samples)
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     samples_orig_values,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                (should_accept,
                 num_tests_needed) = cls.accept_attribute(samples_orig_values,
                                                          curr_gain,
                                                          num_tests,
                                                          tree_node.dataset.num_classes,
                                                          tree_node.valid_samples_indices,
                                                          attrib_num_valid_values,
                                                          orig_to_new_value_int,
                                                          new_to_orig_value_int,
                                                          tree_node.class_index_num_samples,
                                                          tree_node.dataset.sample_class,
                                                          diff_keys,
                                                          diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            should_accept,
                            num_tests_needed))

        preference_rank_full = sorted(ret,
                                      key=lambda x: (-x[5], 0, -x[1])
                                      if x[5]
                                      else (-x[5], -x[6], -x[1]))

        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)

        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)

        num_valid_attrib = sum(tree_node.valid_nominal_attribute)
        num_tests = int(math.ceil(math.log2(num_valid_attrib))) + 6

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue

                samples_orig_values = cls._generate_samples_orig_values(
                    attrib_index,
                    tree_node.valid_samples_indices,
                    tree_node.dataset.samples)
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     samples_orig_values,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                should_accept, _ = cls.accept_attribute(samples_orig_values,
                                                        curr_gain,
                                                        num_tests,
                                                        tree_node.dataset.num_classes,
                                                        tree_node.valid_samples_indices,
                                                        attrib_num_valid_values,
                                                        orig_to_new_value_int,
                                                        new_to_orig_value_int,
                                                        tree_node.class_index_num_samples,
                                                        tree_node.dataset.sample_class,
                                                        diff_keys,
                                                        diff_values)
                if not should_accept:
                    continue
                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @staticmethod
    def _generate_samples_orig_values(attrib_index, valid_samples_indices, samples):
        #TESTED!
        samples_orig_values = [0] * len(samples)
        for sample_index in valid_samples_indices:
            samples_orig_values[sample_index] = samples[sample_index][attrib_index]
        return samples_orig_values

    @classmethod
    def _generate_best_split(cls, num_classes, attrib_num_valid_values, orig_to_new_value_int,
                             new_to_orig_value_int, valid_samples_indices, class_index_num_samples,
                             samples_orig_values, sample_class, diff_keys, diff_values):

        def _init_values_histograms(num_classes, attrib_num_valid_values, valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples_orig_values[sample_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j:
                        continue
                    for class_index in range(num_classes):
                        num_elems_value_j_diff_class = (
                            values_histogram[value_index_j]
                            - values_histogram_with_classes[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            values_histogram_with_classes[value_index_i, class_index]
                            * num_elems_value_j_diff_class)
            return weights

        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        left_values, right_values = cls._generate_random_partition(frac_split_cholesky,
                                                                   new_to_orig_value_int)
        gain = cls._calculate_split_gain(num_classes,
                                         len(valid_samples_indices),
                                         class_index_num_samples,
                                         sample_class,
                                         samples_orig_values,
                                         right_values,
                                         diff_keys,
                                         diff_values)
        return gain, values_histogram, left_values, right_values

    @staticmethod
    def _calculate_split_gain(num_classes, num_samples, class_index_num_samples, sample_class,
                              samples_orig_values, right_values, diff_keys, diff_values):
        #TESTED!
        def _init_num_samples_right_split_and_tcv(num_classes, sample_class, samples_orig_values,
                                                  right_values, diff_keys):
            #TESTED!
            tcv = np.zeros((num_classes, 2), dtype=np.int64)
            # first column = left/false in values_split
            num_samples_right_split = 0
            # tcv[class_index][0] is for samples on the left side of split and tcv[class_index][1]
            # is for samples on the right side.
            for int_key in diff_keys:
                curr_sample_class = sample_class[int_key]
                sample_int_value = samples_orig_values[int_key]
                if sample_int_value in right_values:
                    num_samples_right_split += 1
                    tcv[curr_sample_class][1] += 1
                else:
                    tcv[curr_sample_class][0] += 1
            return num_samples_right_split, tcv


        # Initialize auxiliary variables
        gain = 0.0
        tc = class_index_num_samples[:] # this slice makes a copy of class_index_num_samples
        num_samples_right_split, tcv = _init_num_samples_right_split_and_tcv(num_classes,
                                                                             sample_class,
                                                                             samples_orig_values,
                                                                             right_values,
                                                                             diff_keys)
        # Calculate gain and update auxiliary variables

        # Samples we haven't dealt with yet, including the current one. Will subtract 1 at every
        # loop, including first.
        num_remaining_samples = num_samples + 1
        for int_key, sample_diff in zip(diff_keys, diff_values):
            curr_sample_class = sample_class[int_key]
            sample_atrib_int_value = samples_orig_values[int_key]

            num_remaining_samples -= 1
            num_elems_in_compl_tc = num_remaining_samples - tc[curr_sample_class]

            # Let's calculate the number of samples in same split side (not yet seen in loop) with
            # different class.
            if sample_atrib_int_value in right_values:
                num_elems_compl_tc_same_split = num_samples_right_split - tcv[curr_sample_class][1]
            else:
                num_samples_left_split = num_remaining_samples - num_samples_right_split
                num_elems_compl_tc_same_split = num_samples_left_split - tcv[curr_sample_class][0]

            gain += sample_diff * (num_elems_in_compl_tc - num_elems_compl_tc_same_split)

            # Time to update the auxiliary variables. We decrement tc and tcv so they only have
            # information concerning samples not yet seen in this for loop.
            tc[curr_sample_class] -= 1
            if sample_atrib_int_value in right_values:
                tcv[curr_sample_class][1] -= 1
                num_samples_right_split -= 1
            else:
                tcv[curr_sample_class][0] -= 1
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set([])
        right_values = set([])
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
            else:
                right_values.add(new_to_orig_value_int[new_value])

        return left_values, right_values

    @staticmethod
    def get_values_dist(samples_orig_values, valid_samples_indices):
        # TESTED!
        seen_values_count = {}
        max_value_seen = 0
        for sample_index in valid_samples_indices:
            sample_value = samples_orig_values[sample_index]
            if sample_value not in seen_values_count:
                seen_values_count[sample_value] = 1
            else:
                seen_values_count[sample_value] += 1
            if sample_value > max_value_seen:
                max_value_seen = sample_value
        values_dist = [0.0] * (max_value_seen + 1)
        num_items = len(valid_samples_indices)
        for seen_value, count in seen_values_count.items():
            values_dist[seen_value] = float(count)/float(num_items)
        return values_dist

    @staticmethod
    def generate_random_values(values_dist, valid_samples_indices, total_samples):
        # TESTED!
        random_values = np.random.choice(len(values_dist),
                                         len(valid_samples_indices),
                                         replace=True,
                                         p=values_dist)
        ret = [-1] * total_samples
        for sample_index, random_value in zip(valid_samples_indices, random_values):
            ret[sample_index] = random_value
        return ret

    @classmethod
    def accept_attribute(cls, samples_orig_values, real_gain, num_tests, num_classes,
                         valid_samples_indices, attrib_num_valid_values, orig_to_new_value_int,
                         new_to_orig_value_int, class_index_num_samples, sample_class, diff_keys,
                         diff_values):
        # TESTED!
        values_dist = cls.get_values_dist(samples_orig_values, valid_samples_indices)
        for test_number in range(1, num_tests + 1):
            random_values = cls.generate_random_values(values_dist,
                                                       valid_samples_indices,
                                                       len(samples_orig_values))
            gain, _, _, _ = cls._generate_best_split(num_classes,
                                                     attrib_num_valid_values,
                                                     orig_to_new_value_int,
                                                     new_to_orig_value_int,
                                                     valid_samples_indices,
                                                     class_index_num_samples,
                                                     random_values,
                                                     sample_class,
                                                     diff_keys,
                                                     diff_values)
            if gain > real_gain:
                return False, test_number
        return True, None



#################################################################################################
#################################################################################################
###                                                                                           ###
###                               MAX CUT MONTE CARLO RESIDUE                                 ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class MaxCutMonteCarloResidue(Criterion):
    name = 'Max Cut Monte Carlo Residue'


    @classmethod
    def evaluate_all_attributes(cls, tree_node):
        #TESTED!

        ret = [] # contains (attrib_index, gain_ratio, split_values, p_value, time_taken,
                 #           should_accept, num_tests_needed)
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)
        num_valid_attrib = sum(tree_node.valid_nominal_attribute)
        num_tests = int(math.ceil(math.log2(num_valid_attrib))) + 6

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                start_time = timeit.default_timer()
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue
                samples_orig_values = cls._generate_samples_orig_values(
                    attrib_index,
                    tree_node.valid_samples_indices,
                    tree_node.dataset.samples)
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     samples_orig_values,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                (should_accept,
                 num_tests_needed) = cls.accept_attribute(samples_orig_values,
                                                          curr_gain,
                                                          num_tests,
                                                          tree_node.dataset.num_classes,
                                                          tree_node.valid_samples_indices,
                                                          attrib_num_valid_values,
                                                          orig_to_new_value_int,
                                                          new_to_orig_value_int,
                                                          tree_node.class_index_num_samples,
                                                          tree_node.dataset.sample_class,
                                                          diff_keys,
                                                          diff_values)
                ret.append((attrib_index,
                            curr_gain,
                            [left_int_values, right_int_values],
                            None,
                            timeit.default_timer() - start_time,
                            should_accept,
                            num_tests_needed))

        preference_rank_full = sorted(ret,
                                      key=lambda x: (-x[5], 0, -x[1])
                                      if x[5]
                                      else (-x[5], -x[6], -x[1]))

        seen_attrib = [False] * len(tree_node.dataset.attrib_names)
        preference_rank = []
        for pref_elem in preference_rank_full:
            if seen_attrib[pref_elem[0]]:
                continue
            seen_attrib[pref_elem[0]] = True
            preference_rank.append(pref_elem)

        ret_with_preference_full = [0] * len(tree_node.dataset.attrib_names)
        for preference, elem in enumerate(preference_rank):
            attrib_index = elem[0]
            new_elem = list(elem)
            new_elem.append(preference)
            ret_with_preference_full[attrib_index] = tuple(new_elem)
        ret_with_preference = [elem for elem in ret_with_preference_full if elem != 0]

        return ret_with_preference

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        best_attrib_index = 0
        best_gain = float('-inf')
        best_split_left_values = set([])
        best_split_right_values = set([])
        diff_keys, diff_values = cls._calculate_diff(tree_node.valid_samples_indices,
                                                     tree_node.dataset.sample_costs)

        num_valid_attrib = sum(tree_node.valid_nominal_attribute)
        num_tests = int(math.ceil(math.log2(num_valid_attrib))) + 6

        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                (attrib_num_valid_values,
                 orig_to_new_value_int,
                 new_to_orig_value_int) = cls._get_attrib_valid_values(
                     attrib_index,
                     tree_node.dataset.samples,
                     tree_node.valid_samples_indices)
                if attrib_num_valid_values <= 1:
                    print("Attribute {} ({}) is valid but has only {} value(s).".format(
                        attrib_index,
                        tree_node.dataset.attrib_names[attrib_index],
                        attrib_num_valid_values))
                    continue

                samples_orig_values = cls._generate_samples_orig_values(
                    attrib_index,
                    tree_node.valid_samples_indices,
                    tree_node.dataset.samples)
                (curr_gain,
                 _,
                 left_int_values,
                 right_int_values) = cls._generate_best_split(
                     tree_node.dataset.num_classes,
                     attrib_num_valid_values,
                     orig_to_new_value_int,
                     new_to_orig_value_int,
                     tree_node.valid_samples_indices,
                     tree_node.class_index_num_samples,
                     samples_orig_values,
                     tree_node.dataset.sample_class,
                     diff_keys,
                     diff_values)
                should_accept, _ = cls.accept_attribute(samples_orig_values,
                                                        curr_gain,
                                                        num_tests,
                                                        tree_node.dataset.num_classes,
                                                        tree_node.valid_samples_indices,
                                                        attrib_num_valid_values,
                                                        orig_to_new_value_int,
                                                        new_to_orig_value_int,
                                                        tree_node.class_index_num_samples,
                                                        tree_node.dataset.sample_class,
                                                        diff_keys,
                                                        diff_values)
                if not should_accept:
                    continue
                if curr_gain > best_gain:
                    best_attrib_index = attrib_index
                    best_gain = curr_gain
                    best_split_left_values = left_int_values
                    best_split_right_values = right_int_values
        splits_values = [best_split_left_values, best_split_right_values]
        return (best_attrib_index, splits_values, best_gain, None)

    @staticmethod
    def _get_attrib_valid_values(attrib_index, samples, valid_samples_indices):
        #TESTED!
        seen_values = set([])
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for sample_index in valid_samples_indices:
            value_int = samples[sample_index][attrib_index]
            if value_int not in seen_values:
                orig_to_new_value_int[value_int] = len(seen_values)
                new_to_orig_value_int.append(value_int)
                seen_values.add(value_int)
        return len(seen_values), orig_to_new_value_int, new_to_orig_value_int

    @staticmethod
    def _calculate_diff(valid_samples_indices, sample_costs):
        #TESTED!
        def _max_min_diff(list_of_values):
            max_val = list_of_values[0]
            min_val = max_val
            for value in list_of_values[1:]:
                if value > max_val:
                    max_val = value
                elif value < min_val:
                    min_val = value
            return abs(max_val - min_val)

        diff_keys = []
        diff_values = []
        for sample_index in valid_samples_indices:
            curr_costs = sample_costs[sample_index]
            diff_values.append(_max_min_diff(curr_costs))
            diff_keys.append(sample_index)
        diff_keys_values = sorted(list(zip(diff_keys, diff_values)),
                                  key=lambda key_value: key_value[1])
        diff_keys, diff_values = zip(*diff_keys_values)
        return diff_keys, diff_values

    @staticmethod
    def _generate_samples_orig_values(attrib_index, valid_samples_indices, samples):
        #TESTED!
        samples_orig_values = [0] * len(samples)
        for sample_index in valid_samples_indices:
            samples_orig_values[sample_index] = samples[sample_index][attrib_index]
        return samples_orig_values

    @classmethod
    def _generate_best_split(cls, num_classes, attrib_num_valid_values, orig_to_new_value_int,
                             new_to_orig_value_int, valid_samples_indices, class_index_num_samples,
                             samples_orig_values, sample_class, diff_keys, diff_values):

        def _init_values_histograms(num_classes, attrib_num_valid_values, valid_samples_indices):
            #TESTED!
            values_histogram = np.zeros((attrib_num_valid_values), dtype=np.int64)
            values_histogram_with_classes = np.zeros((attrib_num_valid_values, num_classes),
                                                     dtype=np.int64)
            for sample_index in valid_samples_indices:
                orig_value = samples_orig_values[sample_index]
                new_value = orig_to_new_value_int[orig_value]
                values_histogram[new_value] += 1
                values_histogram_with_classes[new_value][sample_class[sample_index]] += 1
            return values_histogram, values_histogram_with_classes

        def _init_values_weights(num_classes, values_histogram, values_histogram_with_classes):
            # TESTED!
            # Initializes the weight of each edge in the values graph (to be sent to the Max Cut)
            weights = np.zeros((values_histogram.shape[0], values_histogram.shape[0]),
                               dtype=np.float64)
            for value_index_i in range(values_histogram.shape[0]):
                for value_index_j in range(values_histogram.shape[0]):
                    if value_index_i == value_index_j:
                        continue

                    mixed_class_dist = np.add(values_histogram_with_classes[value_index_i],
                                              values_histogram_with_classes[value_index_j])

                    total_elems_value_pair = (values_histogram[value_index_i]
                                              + values_histogram[value_index_j])
                    left_frac = values_histogram[value_index_i] / total_elems_value_pair
                    right_frac = values_histogram[value_index_j] / total_elems_value_pair

                    for class_index in range(num_classes):
                        num_elems_value_j_diff_class = (
                            values_histogram[value_index_j]
                            - values_histogram_with_classes[value_index_j, class_index])
                        weights[value_index_i, value_index_j] += (
                            values_histogram_with_classes[value_index_i, class_index]
                            * num_elems_value_j_diff_class)

                        # Let's subtract the average cut for this pair of values with this
                        # distribution.
                        weights[value_index_i, value_index_j] -= (
                            left_frac * mixed_class_dist[class_index] *
                            (values_histogram[value_index_j]
                             - right_frac * mixed_class_dist[class_index]))

                    if weights[value_index_i, value_index_j] < 0.0:
                        weights[value_index_i, value_index_j] = 0.0

            return weights

        (values_histogram,
         values_histogram_with_classes) = _init_values_histograms(num_classes,
                                                                  attrib_num_valid_values,
                                                                  valid_samples_indices)
        weights = _init_values_weights(num_classes,
                                       values_histogram,
                                       values_histogram_with_classes)

        frac_split_cholesky = cls._solve_max_cut(attrib_num_valid_values, weights)
        if frac_split_cholesky is None:
            return None, None, None, None
        left_values, right_values = cls._generate_random_partition(frac_split_cholesky,
                                                                   new_to_orig_value_int)
        gain = cls._calculate_split_gain(num_classes,
                                         len(valid_samples_indices),
                                         class_index_num_samples,
                                         sample_class,
                                         samples_orig_values,
                                         right_values,
                                         diff_keys,
                                         diff_values)
        return gain, values_histogram, left_values, right_values

    @staticmethod
    def _calculate_split_gain(num_classes, num_samples, class_index_num_samples, sample_class,
                              samples_orig_values, right_values, diff_keys, diff_values):
        #TESTED!
        def _init_num_samples_right_split_and_tcv(num_classes, sample_class, samples_orig_values,
                                                  right_values, diff_keys):
            #TESTED!
            tcv = np.zeros((num_classes, 2), dtype=np.int64)
            # first column = left/false in values_split
            num_samples_right_split = 0
            # tcv[class_index][0] is for samples on the left side of split and tcv[class_index][1]
            # is for samples on the right side.
            for int_key in diff_keys:
                curr_sample_class = sample_class[int_key]
                sample_int_value = samples_orig_values[int_key]
                if sample_int_value in right_values:
                    num_samples_right_split += 1
                    tcv[curr_sample_class][1] += 1
                else:
                    tcv[curr_sample_class][0] += 1
            return num_samples_right_split, tcv


        # Initialize auxiliary variables
        gain = 0.0
        tc = class_index_num_samples[:] # this slice makes a copy of class_index_num_samples
        num_samples_right_split, tcv = _init_num_samples_right_split_and_tcv(num_classes,
                                                                             sample_class,
                                                                             samples_orig_values,
                                                                             right_values,
                                                                             diff_keys)
        # Calculate gain and update auxiliary variables

        # Samples we haven't dealt with yet, including the current one. Will subtract 1 at every
        # loop, including first.
        num_remaining_samples = num_samples + 1
        for int_key, sample_diff in zip(diff_keys, diff_values):
            curr_sample_class = sample_class[int_key]
            sample_atrib_int_value = samples_orig_values[int_key]

            num_remaining_samples -= 1
            num_elems_in_compl_tc = num_remaining_samples - tc[curr_sample_class]

            # Let's calculate the number of samples in same split side (not yet seen in loop) with
            # different class.
            if sample_atrib_int_value in right_values:
                num_elems_compl_tc_same_split = num_samples_right_split - tcv[curr_sample_class][1]
            else:
                num_samples_left_split = num_remaining_samples - num_samples_right_split
                num_elems_compl_tc_same_split = num_samples_left_split - tcv[curr_sample_class][0]

            gain += sample_diff * (num_elems_in_compl_tc - num_elems_compl_tc_same_split)

            # Time to update the auxiliary variables. We decrement tc and tcv so they only have
            # information concerning samples not yet seen in this for loop.
            tc[curr_sample_class] -= 1
            if sample_atrib_int_value in right_values:
                tcv[curr_sample_class][1] -= 1
                num_samples_right_split -= 1
            else:
                tcv[curr_sample_class][0] -= 1
        return gain

    @staticmethod
    def _solve_max_cut(attrib_num_valid_values, weights):
        #TESTED!
        def _solve_sdp(size, weights):
            #TESTED!
            # See Max Cut approximate given by Goemans and Williamson, 1995.
            var = cvx.Semidef(size)
            obj = cvx.Minimize(0.25 * cvx.trace(weights.T * var))

            constraints = [var == var.T, var >> 0]
            for i in range(size):
                constraints.append(var[i, i] == 1)

            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.SCS, verbose=False)
            return var.value

        fractional_split_squared = _solve_sdp(attrib_num_valid_values, weights)
        if fractional_split_squared is None:
            return None
        # The solution should be symmetric, but let's just make sure the approximations didn't
        # change that.
        sym_fractional_split_squared = 0.5 * (fractional_split_squared
                                              + fractional_split_squared.T)
        # We are interested in the Cholesky decomposition of the above matrix to finally choose a
        # random partition based on it. Detail: the above matrix may be singular, so not every
        # method works.
        temp_P, temp_L, _ = chol.chol_higham(sym_fractional_split_squared)

        # Note that temp_L.T is upper triangular, but
        # frac_split_cholesky = np.dot(temp.L.T, temp_P)
        # is not necessarily upper triangular. Since we are only interested in decomposing
        # sym_fractional_split_squared = np.dot(frac_split_cholesky.T, frac_split_cholesky)
        # that is not a problem.
        return np.dot(temp_L.T, temp_P)

    @staticmethod
    def _generate_random_partition(frac_split_cholesky,
                                   new_to_orig_value_int):
        #TESTED!
        random_vector = np.random.randn(frac_split_cholesky.shape[1])
        values_split = np.zeros((frac_split_cholesky.shape[1]), dtype=np.float64)
        for column_index in range(frac_split_cholesky.shape[1]):
            column = frac_split_cholesky[:, column_index]
            values_split[column_index] = np.dot(random_vector, column)
        values_split_bool = np.apply_along_axis(lambda x: x > 0.0, axis=0, arr=values_split)
        # Let's get the values on each side of this partition
        left_values = set([])
        right_values = set([])
        for new_value in range(frac_split_cholesky.shape[1]):
            if values_split_bool[new_value]:
                left_values.add(new_to_orig_value_int[new_value])
            else:
                right_values.add(new_to_orig_value_int[new_value])

        return left_values, right_values

    @staticmethod
    def get_values_dist(samples_orig_values, valid_samples_indices):
        # TESTED!
        seen_values_count = {}
        max_value_seen = 0
        for sample_index in valid_samples_indices:
            sample_value = samples_orig_values[sample_index]
            if sample_value not in seen_values_count:
                seen_values_count[sample_value] = 1
            else:
                seen_values_count[sample_value] += 1
            if sample_value > max_value_seen:
                max_value_seen = sample_value
        values_dist = [0.0] * (max_value_seen + 1)
        num_items = len(valid_samples_indices)
        for seen_value, count in seen_values_count.items():
            values_dist[seen_value] = float(count)/float(num_items)
        return values_dist

    @staticmethod
    def generate_random_values(values_dist, valid_samples_indices, total_samples):
        # TESTED!
        random_values = np.random.choice(len(values_dist),
                                         len(valid_samples_indices),
                                         replace=True,
                                         p=values_dist)
        ret = [-1] * total_samples
        for sample_index, random_value in zip(valid_samples_indices, random_values):
            ret[sample_index] = random_value
        return ret

    @classmethod
    def accept_attribute(cls, samples_orig_values, real_gain, num_tests, num_classes,
                         valid_samples_indices, attrib_num_valid_values, orig_to_new_value_int,
                         new_to_orig_value_int, class_index_num_samples, sample_class, diff_keys,
                         diff_values):
        # TESTED!
        values_dist = cls.get_values_dist(samples_orig_values, valid_samples_indices)
        for test_number in range(1, num_tests + 1):
            restart = True
            while restart:
                random_values = cls.generate_random_values(values_dist,
                                                           valid_samples_indices,
                                                           len(samples_orig_values))
                gain, _, _, _ = cls._generate_best_split(num_classes,
                                                         attrib_num_valid_values,
                                                         orig_to_new_value_int,
                                                         new_to_orig_value_int,
                                                         valid_samples_indices,
                                                         class_index_num_samples,
                                                         random_values,
                                                         sample_class,
                                                         diff_keys,
                                                         diff_values)
                if gain is not None:
                    restart = False
            if gain > real_gain:
                return False, test_number
        return True, None
