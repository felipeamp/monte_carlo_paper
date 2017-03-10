#!/usr/bin/python3
# -*- coding: utf-8 -*-


import datetime
import itertools
import math
import os
import random
import sys

import criteria
import dataset
import decision_tree

import numpy as np
from scipy.special import binom as binom_coef
from scipy.stats import binom as binom_stats


RANDOM_SEED = 65537
OUTPUT_SPLIT_CHAR = ','

# TABLE_T_F_M[t][f][m] = E, where data structure before [m] is a list and TABLE_T_F_M[t][f][m]
# gives us E for m+1 attributes.
TABLE_T_F_M = {}
# TABLE_U_L_P_M[u][l][p] = (t, f)
TABLE_U_L_P_M = {}


def get_expected_by_attrib(u_value, l_value, prob_value, m):
    def save_t_f_m(t, f, m, E):
        if t not in TABLE_T_F_M:
            TABLE_T_F_M[t] = {}
        if f not in TABLE_T_F_M[t]:
            TABLE_T_F_M[t][f] = []
        if len(TABLE_T_F_M[t][f]) != m - 1:
            print("Wrong length of TABLE_T_F_M[t][f] when saving m. (t, f, m) =", (t, f, m))
            sys.exit(1)
        else:
            TABLE_T_F_M[t][f].append(E)

    def get_t_f_m(t, f, m):
        if (t not in TABLE_T_F_M
                or f not in TABLE_T_F_M[t]
                or len(TABLE_T_F_M[t][f]) < m):
            return None
        return TABLE_T_F_M[t][f][m-1]


    t, f = get_t_f(u_value, l_value, prob_value, m)
    if t is None or f is None:
        return None, None, None

    # DEBUG:
    print('t found =', t)
    print('f found =', f)

    E = get_t_f_m(t, f, m)
    if E is not None:
        return t, f, E

    s_array = get_s_coef_array(t, f)
    ds_array = get_s_derivative_coef_array(t, f)
    max_value_for_root = get_max_value_for_root(t, f)

    if t in TABLE_T_F_M and f in TABLE_T_F_M[t]:
        starting_m = len(TABLE_T_F_M[t][f]) + 1
        o_i = TABLE_T_F_M[t][f][-1]
    else:
        o_i = get_o_1(t)#, f, s_array, ds_array, max_value_for_root)
        print('o_1 =', o_i)
        save_t_f_m(t, f, 1, o_i)
        starting_m = 2

    for i in range(starting_m, m + 1):
        print('='*80)
        print('i =', i)
        r_i_prev = get_r_i(t, f, s_array, ds_array, o_i, max_value_for_root)
        # DEBUG
        print('r_{i-1}:')
        print(r_i_prev)
        print('*'*50)
        print('Calculating possible increments:')
        max_increment_found = max(get_increment_o_i(q, t, f, o_i) for q in r_i_prev)
        o_i += max_increment_found
        print('*'*50)
        print('max_increment_found =', max_increment_found)
        print('o_i =', o_i)
        save_t_f_m(t, f, i, o_i)

    # DEBUG
    print('='*80)
    print('t =', t)
    print('f =', f)
    print('m =', m)
    print('o_m =', o_i)
    print('E =', o_i)
    print('E/m =', o_i/m)
    print('='*80)

    return t, f, o_i


def get_t_f(u_value, l_value, prob_value, num_attrib):
    def is_ok_l(t, f, l_value, prob_value):
        return get_s(1 - l_value, t, f) >= prob_value

    def is_ok_u(t, f, u_value, prob_value, num_attrib):
        return get_s(1 - u_value, t, f) <= 1. - math.pow(prob_value, 1. / num_attrib)

    def save_u_l_p_m(u_value, l_value, prob_value, num_attrib, t, f):
        if u_value not in TABLE_U_L_P_M:
            TABLE_U_L_P_M[u_value] = {}
        if l_value not in TABLE_U_L_P_M[u_value]:
            TABLE_U_L_P_M[u_value][l_value] = {}
        if prob_value not in TABLE_U_L_P_M[u_value][l_value]:
            TABLE_U_L_P_M[u_value][l_value][prob_value] = {}
        if num_attrib not in TABLE_U_L_P_M[u_value][l_value][prob_value]:
            TABLE_U_L_P_M[u_value][l_value][prob_value][num_attrib] = (t, f)

    def get_u_l_p_m(u_value, l_value, prob_value, num_attrib):
        if (u_value not in TABLE_U_L_P_M
                or l_value not in TABLE_U_L_P_M[u_value]
                or prob_value not in TABLE_U_L_P_M[u_value][l_value]
                or num_attrib not in TABLE_U_L_P_M[u_value][l_value][prob_value]):
            return (None, None)
        return TABLE_U_L_P_M[u_value][l_value][prob_value][num_attrib]


    (t, f) = get_u_l_p_m(u_value, l_value, prob_value, num_attrib)
    if t is not None and f is not None:
        return (t, f)

    for f in range(1, 501):
        # Let's get the largest t that satisfies the condition for L.
        # Note that, since s(x, t, f) decreases as t increases, all t smaller than the one we'll
        # find satisty the condition for L.
        t_low = f + 1
        t_high = 1000
        print('-'*80)
        print('f =', f)
        print('t_low =', t_low)
        print('t_high =', t_high)
        while t_high > t_low:
            t_mid = (t_low + t_high + 1) // 2 # Round up
            if is_ok_l(t_mid, f, l_value, prob_value):
                t_low = t_mid
                print('t_low =', t_low)
                print('t_high =', t_high)
            else:
                t_high = t_mid - 1
                print('t_low =', t_low)
                print('t_high =', t_high)
        if not is_ok_l(t_high, f, l_value, prob_value):
            continue
        else:
            largest_t_l = t_high

        # Let's get the smallest t that satisfies the condition for U.
        # Note that, since s(x, t, f) decreases as t increases, all t larger than the one we'll
        # find satisty the condition for U.
        t_low = f + 1
        t_high = 1000
        print('-'*80)
        print('f =', f)
        print('t_low =', t_low)
        print('t_high =', t_high)
        while t_high > t_low:
            t_mid = (t_low + t_high) // 2
            if is_ok_u(t_mid, f, u_value, prob_value, num_attrib):
                t_high = t_mid
                print('t_low =', t_low)
                print('t_high =', t_high)
            else:
                t_low = t_mid + 1
                print('t_low =', t_low)
                print('t_high =', t_high)
        if not is_ok_u(t_high, f, u_value, prob_value, num_attrib):
            continue
        else:
            smallest_t_u = t_high

        if smallest_t_u <= largest_t_l:
            save_u_l_p_m(u_value, l_value, prob_value, num_attrib, smallest_t_u, f)
            return smallest_t_u, f
    return None, None


def get_o_1(t):
    return t


def get_max_value_for_root(t, f):
    return 1. - f / t


def get_s_derivative_coef_array(t, f):
    ret = np.zeros(t + 1, dtype=float)
    for j in range(0, f):
        curr_coef = 0.0
        for i in range(j, f):
            if (i - j) & 1: # (i - j) is odd, thus (-1)**(i-j) == -1
                curr_coef -= binom_coef(t, i) * binom_coef(i, j)
            else: # (i - j) is even, thus (-1)**(i-j) == 1
                curr_coef += binom_coef(t, i) * binom_coef(i, j)
        ret[j + 1] = (t - j) * curr_coef
    return ret


def get_s_coef_array(t, f):
    ret = np.zeros(t + 1, dtype=float)
    for k in range(0, f):
        curr_coef = 0.0
        for i in range(k, f):
            if (i - k) & 1: # (i - k) is odd, thus (-1)**(i-k) == -1
                curr_coef -= binom_coef(t, i) * binom_coef(i, k)
            else: # (i - k) is even, thus (-1)**(i-k) == 1
                curr_coef += binom_coef(t, i) * binom_coef(i, k)
        ret[k] = curr_coef
    return ret


def get_r_i(t, f, s_array, ds_array, o_i, max_value_for_root):
    coef_array = np.zeros(2 + s_array.shape[0], dtype=float)

    # coef_array[2:] += ds_array
    # coef_array[1:-1] += -2.0 * ds_array
    # coef_array[:-2] += ds_array
    # coef_array[-1] -= f / o_i

    coef_array[2:] += ds_array * (t - o_i - f) - s_array * f
    coef_array[1:-1] += ds_array * (2.0 * (o_i - t) + f)
    coef_array[:-2] += ds_array * (t - o_i)
    coef_array[-1] += f

    roots = get_poly_roots_in_interval(coef_array, 0.0, max_value_for_root)
    roots.add(0.0)
    roots.add(max_value_for_root)

    # DEBUG
    if len(roots) > 2:
        print('==*=='*20)
        print('\t\tFound {} roots!'.format(len(roots) - 2))
        print('\t\tIncluding the interval boundaries, we have')
        print('\t\troots =', roots)
        print('==*=='*20)
    return roots

def get_poly_roots_in_interval(coef_array, min_value_for_root, max_value_for_root):
    # # DEBUG
    # print('coef_array:')
    # print(coef_array)
    roots = np.roots(coef_array)
    # # DEBUG
    # print('roots:')
    # print(roots)
    ret = set()
    for is_real, root in zip(np.isreal(roots), roots):
        if is_real and root.real >= min_value_for_root and root.real <= max_value_for_root:
            ret.add(float(root.real))
    return ret


def get_increment_o_i(q, t, f, o_i):
    s = get_s(q, t, f)
    ret = f / (1. - q) + s * (t - o_i - f / (1. - q))
    # DEBUG
    print('-'*40)
    print('q =', q)
    print('t =', t)
    print('f =', f)
    print('o_{i-1} =', o_i)
    print('s =', s)
    print('increment_o_i =', ret)
    return ret


def get_s(p, t, f):
    return binom_stats.cdf(f - 1, t, 1. - p)




def monte_carlo_tests_numbers(dataset_name, train_dataset, criterion, num_samples, num_trials, U, L,
                              p_L, output_file_descriptor, output_split_char, seed=RANDOM_SEED):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    training_samples_indices = list(range(train_dataset.num_samples))

    tests_done_ordered_list = []
    total_tests_done_ordered = 0
    max_tests_done_ordered = - sys.maxsize
    min_tests_done_ordered = sys.maxsize
    total_tests_per_attrib_ordered = 0
    max_tests_per_attrib_ordered = - sys.maxsize
    min_tests_per_attrib_ordered = sys.maxsize

    tests_done_rev_list = []
    total_tests_done_rev = 0
    max_tests_done_rev = - sys.maxsize
    min_tests_done_rev = sys.maxsize
    total_tests_per_attrib_rev = 0
    max_tests_per_attrib_rev = - sys.maxsize
    min_tests_per_attrib_rev = sys.maxsize

    tests_done_random_order_list = []
    total_tests_done_random_order = 0
    max_tests_done_random_order = - sys.maxsize
    min_tests_done_random_order = sys.maxsize
    total_tests_per_attrib_random_order = 0
    max_tests_per_attrib_random_order = - sys.maxsize
    min_tests_per_attrib_random_order = sys.maxsize

    total_num_valid_attributes = 0
    max_num_valid_attributes = - sys.maxsize
    min_num_valid_attributes = sys.maxsize

    theoretical_e_over_m_list = []
    total_theoretical_e_over_m = 0.0
    max_theoretical_e_over_m = float('-inf')
    min_theoretical_e_over_m = float('inf')

    total_num_tests = 0.0
    max_num_tests = - sys.maxsize
    min_num_tests = sys.maxsize

    total_num_fails_allowed = 0.0
    max_num_fails_allowed = - sys.maxsize
    min_num_fails_allowed = sys.maxsize

    rank_ordered_list = []

    time_taken_criterion_list = []
    time_taken_ordered_list = []
    time_taken_rev_list = []
    time_taken_random_list = []

    num_best_found = 0
    num_best_only_missing = 0
    total_accuracy_sum_best_according_to_criterion_w_missing_values = 0.0
    total_accuracy_sum_best_according_to_criterion_wo_missing_values = 0.0

    num_accepted = 0
    num_accepted_only_missing = 0
    total_accuracy_sum_accepted_w_missing_values = 0.0
    total_accuracy_sum_accepted_wo_missing_values = 0.0

    total_num_samples_missing_values_best_according_to_criterion = 0
    total_num_samples_missing_values_accepted = 0

    for _ in range(num_trials):
        random.shuffle(training_samples_indices)
        curr_training_samples_indices = training_samples_indices[:num_samples]
        curr_test_samples_indices = training_samples_indices[num_samples: 2 * num_samples]
        node = decision_tree.TreeNode(train_dataset,
                                      curr_training_samples_indices,
                                      train_dataset.valid_nominal_attribute[:],
                                      max_depth_remaining=1,
                                      min_samples_per_node=1)

        while sorted(node.class_index_num_samples)[-2] == 0:
            random.shuffle(training_samples_indices)
            curr_training_samples_indices = training_samples_indices[:num_samples]
            curr_test_samples_indices = training_samples_indices[num_samples: 2 * num_samples]
            node = decision_tree.TreeNode(train_dataset,
                                          curr_training_samples_indices,
                                          train_dataset.valid_nominal_attribute[:],
                                          max_depth_remaining=1,
                                          min_samples_per_node=1)

        # Calculate the number of valid atributes with at least 2 different values
        num_valid_attributes = 0
        for (attrib_index,
             is_valid_nominal_attrib) in enumerate(train_dataset.valid_nominal_attribute):
            if is_valid_nominal_attrib:
                values_seen = get_values_seen(node.contingency_tables[attrib_index][1])
                if len(values_seen) > 1:
                    num_valid_attributes += 1


        t, f, E = get_expected_by_attrib(U, L, p_L, num_valid_attributes)
        theoretical_e_over_m = E / num_valid_attributes
        num_fails_allowed = f - 1

        (tests_done_ordered,
         _,
         tests_done_rev,
         _,
         tests_done_random_order,
         _,
         _,
         ordered_accepted_rank,
         criterion_total_time,
         ordered_total_time,
         rev_total_time,
         random_total_time,
         best_according_to_criterion,
         accepted) = criterion.evaluate_all_attributes_2(node, t, num_fails_allowed)

        if not math.isinf(best_according_to_criterion[1]):
            num_best_found += 1
            curr_accuracy_best_according_to_criterion = get_accuracy(train_dataset,
                                                                     node.most_common_int_class,
                                                                     curr_training_samples_indices,
                                                                     curr_test_samples_indices,
                                                                     best_according_to_criterion[0],
                                                                     best_according_to_criterion[2])
            total_accuracy_sum_best_according_to_criterion_w_missing_values += (
                curr_accuracy_best_according_to_criterion[0])
            if curr_accuracy_best_according_to_criterion[1] is None:
                num_best_only_missing += 1
            else:
                total_accuracy_sum_best_according_to_criterion_wo_missing_values += (
                    curr_accuracy_best_according_to_criterion[1])
            total_num_samples_missing_values_best_according_to_criterion += (
                curr_accuracy_best_according_to_criterion[2])
        else:
            print('BEST IS INF!')
            sys.exit(1)

        num_accepted += 1
        if accepted is not None:
            curr_accuracy_accepted = get_accuracy(train_dataset,
                                                  node.most_common_int_class,
                                                  curr_training_samples_indices,
                                                  curr_test_samples_indices,
                                                  accepted[0],
                                                  accepted[2])
        else:
            curr_accuracy_accepted = get_accuracy_root(train_dataset,
                                                       node.most_common_int_class,
                                                       curr_test_samples_indices)
        total_accuracy_sum_accepted_w_missing_values += curr_accuracy_accepted[0]
        if curr_accuracy_accepted[1] is None:
            num_accepted_only_missing += 1
        else:
            total_accuracy_sum_accepted_wo_missing_values += curr_accuracy_accepted[1]
        total_num_samples_missing_values_accepted += curr_accuracy_accepted[2]


        rank_ordered_list.append(ordered_accepted_rank)
        time_taken_criterion_list.append(criterion_total_time)
        time_taken_ordered_list.append(ordered_total_time)
        time_taken_rev_list.append(rev_total_time)
        time_taken_random_list.append(random_total_time)

        tests_done_ordered_list.append(tests_done_ordered)
        tests_done_rev_list.append(tests_done_rev)
        tests_done_random_order_list.append(tests_done_random_order)


        total_tests_done_ordered += tests_done_ordered
        tests_per_attrib_ordered = tests_done_ordered / num_valid_attributes
        total_tests_per_attrib_ordered += tests_per_attrib_ordered
        total_tests_done_rev += tests_done_rev
        tests_per_attrib_rev = tests_done_rev / num_valid_attributes
        total_tests_per_attrib_rev += tests_per_attrib_rev
        total_tests_done_random_order += tests_done_random_order
        tests_per_attrib_random_order = tests_done_random_order / num_valid_attributes
        total_tests_per_attrib_random_order += tests_per_attrib_random_order
        total_num_valid_attributes += num_valid_attributes

        theoretical_e_over_m_list.append(theoretical_e_over_m)
        total_theoretical_e_over_m += theoretical_e_over_m
        total_num_tests += t
        total_num_fails_allowed += num_fails_allowed

        if tests_done_ordered > max_tests_done_ordered:
            max_tests_done_ordered = tests_done_ordered
        if tests_done_ordered < min_tests_done_ordered:
            min_tests_done_ordered = tests_done_ordered
        if tests_per_attrib_ordered > max_tests_per_attrib_ordered:
            max_tests_per_attrib_ordered = tests_per_attrib_ordered
        if tests_per_attrib_ordered < min_tests_per_attrib_ordered:
            min_tests_per_attrib_ordered = tests_per_attrib_ordered

        if tests_done_rev > max_tests_done_rev:
            max_tests_done_rev = tests_done_rev
        if tests_done_rev < min_tests_done_rev:
            min_tests_done_rev = tests_done_rev
        if tests_per_attrib_rev > max_tests_per_attrib_rev:
            max_tests_per_attrib_rev = tests_per_attrib_rev
        if tests_per_attrib_rev < min_tests_per_attrib_rev:
            min_tests_per_attrib_rev = tests_per_attrib_rev

        if tests_done_random_order > max_tests_done_random_order:
            max_tests_done_random_order = tests_done_random_order
        if tests_done_random_order < min_tests_done_random_order:
            min_tests_done_random_order = tests_done_random_order
        if tests_per_attrib_random_order > max_tests_per_attrib_random_order:
            max_tests_per_attrib_random_order = tests_per_attrib_random_order
        if tests_per_attrib_random_order < min_tests_per_attrib_random_order:
            min_tests_per_attrib_random_order = tests_per_attrib_random_order


        if num_valid_attributes > max_num_valid_attributes:
            max_num_valid_attributes = num_valid_attributes
        if num_valid_attributes < min_num_valid_attributes:
            min_num_valid_attributes = num_valid_attributes

        if theoretical_e_over_m > max_theoretical_e_over_m:
            max_theoretical_e_over_m = theoretical_e_over_m
        if theoretical_e_over_m < min_theoretical_e_over_m:
            min_theoretical_e_over_m = theoretical_e_over_m

        if t > max_num_tests:
            max_num_tests = t
        if t < min_num_tests:
            min_num_tests = t

        if num_fails_allowed > max_num_fails_allowed:
            max_num_fails_allowed = num_fails_allowed
        if num_fails_allowed < min_num_fails_allowed:
            min_num_fails_allowed = num_fails_allowed

    sd_theoretical_e_over_m = np.std(np.array(theoretical_e_over_m_list))
    sd_total_tests_done_ordered = np.std(np.array(tests_done_ordered_list))
    sd_total_tests_per_attrib_ordered = np.std(np.array(
        [tests / num_valid_attributes for tests in tests_done_ordered_list]))
    sd_total_tests_done_rev = np.std(np.array(tests_done_rev_list))
    sd_total_tests_per_attrib_rev = np.std(np.array(
        [tests / num_valid_attributes for tests in tests_done_rev_list]))
    sd_total_tests_done_random_order = np.std(np.array(tests_done_random_order_list))
    sd_total_tests_per_attrib_random_order = np.std(np.array(
        [tests / num_valid_attributes for tests in tests_done_random_order_list]))

    rank_ordered_array = np.array([rank for rank in rank_ordered_list if rank is not None])

    average_rank_ordered = np.mean(rank_ordered_array)
    min_rank_ordered = np.amin(rank_ordered_array)
    max_rank_ordered = np.amax(rank_ordered_array)
    sd_rank_ordered = np.std(rank_ordered_array)
    num_rank_found = sum(rank is not None for rank in rank_ordered_list)

    average_time_criterion = np.mean(np.array(time_taken_criterion_list))
    average_time_ordered = np.mean(np.array(time_taken_ordered_list))
    average_time_rev = np.mean(np.array(time_taken_rev_list))
    average_time_random = np.mean(np.array(time_taken_random_list))

    if num_best_found > 0:
        average_accuracy_best_according_to_criterion_w_missing_values = (
            total_accuracy_sum_best_according_to_criterion_w_missing_values / num_best_found)
        if num_best_found == num_best_only_missing:
            average_accuracy_best_according_to_criterion_wo_missing_values = None
        else:
            average_accuracy_best_according_to_criterion_wo_missing_values = (
                total_accuracy_sum_best_according_to_criterion_wo_missing_values / num_best_found)
        average_num_samples_missing_values_best_according_to_criterion = (
            total_num_samples_missing_values_best_according_to_criterion / num_best_found)
    else:
        average_accuracy_best_according_to_criterion_w_missing_values = None
        average_accuracy_best_according_to_criterion_wo_missing_values = None
        average_num_samples_missing_values_best_according_to_criterion = None

    if num_accepted > 0:
        average_accuracy_accepted_w_missing_values = (
            total_accuracy_sum_accepted_w_missing_values / num_accepted)
        if num_accepted == num_accepted_only_missing:
            average_accuracy_accepted_wo_missing_values = None
        else:
            average_accuracy_accepted_wo_missing_values = (
                total_accuracy_sum_accepted_wo_missing_values / num_accepted)
        average_num_samples_missing_values_accepted = (
            total_num_samples_missing_values_accepted / num_accepted)
    else:
        average_accuracy_accepted_w_missing_values = None
        average_accuracy_accepted_wo_missing_values = None
        average_num_samples_missing_values_accepted = None

    save_fold_info(dataset_name, num_trials, criterion.name, num_samples, total_tests_done_ordered,
                   max_tests_done_ordered, min_tests_done_ordered, total_tests_per_attrib_ordered,
                   max_tests_per_attrib_ordered, min_tests_per_attrib_ordered,
                   total_tests_done_rev, max_tests_done_rev, min_tests_done_rev,
                   total_tests_per_attrib_rev, max_tests_per_attrib_rev, min_tests_per_attrib_rev,
                   total_tests_done_random_order, max_tests_done_random_order,
                   min_tests_done_random_order, total_tests_per_attrib_random_order,
                   max_tests_per_attrib_random_order, min_tests_per_attrib_random_order,
                   total_num_tests, max_num_tests, min_num_tests, total_num_fails_allowed,
                   max_num_fails_allowed, min_num_fails_allowed, total_num_valid_attributes,
                   max_num_valid_attributes, min_num_valid_attributes, total_theoretical_e_over_m,
                   max_theoretical_e_over_m, min_theoretical_e_over_m, U, L, p_L, output_split_char,
                   output_file_descriptor, sd_theoretical_e_over_m, sd_total_tests_done_ordered,
                   sd_total_tests_per_attrib_ordered, sd_total_tests_done_rev,
                   sd_total_tests_per_attrib_rev, sd_total_tests_done_random_order,
                   sd_total_tests_per_attrib_random_order, average_rank_ordered, min_rank_ordered,
                   max_rank_ordered, sd_rank_ordered, num_rank_found, average_time_criterion,
                   average_time_ordered, average_time_rev, average_time_random,
                   average_accuracy_best_according_to_criterion_w_missing_values,
                   average_accuracy_accepted_w_missing_values,
                   average_accuracy_best_according_to_criterion_wo_missing_values,
                   average_accuracy_accepted_wo_missing_values,
                   average_num_samples_missing_values_best_according_to_criterion,
                   average_num_samples_missing_values_accepted)


def get_values_seen(values_num_samples):
    values_seen = set()
    for value, num_samples in enumerate(values_num_samples):
        if num_samples > 0:
            values_seen.add(value)
    return values_seen


def get_accuracy(train_dataset, root_class, training_samples_indices, test_samples_indices,
                 attribute_index, split_values):
    set_left_values = split_values[0]
    set_right_values = split_values[1]

    left_class_index_num_samples = [0] * train_dataset.num_classes
    right_class_index_num_samples = [0] * train_dataset.num_classes
    for training_sample_index in training_samples_indices:
        curr_sample_value = train_dataset.samples[training_sample_index][attribute_index]
        curr_sample_class = train_dataset.sample_class[training_sample_index]
        if curr_sample_value in set_left_values:
            left_class_index_num_samples[curr_sample_class] += 1
        elif curr_sample_value in set_right_values:
            right_class_index_num_samples[curr_sample_class] += 1
        else:
            print('Training sample has unkown value. This shouldn\'t happen.')
            sys.exit(1)
    left_class = left_class_index_num_samples.index(max(left_class_index_num_samples))
    right_class = right_class_index_num_samples.index(max(right_class_index_num_samples))

    num_correct_w_missing = 0
    num_correct_wo_missing = 0
    num_missing = 0
    for test_sample_index in test_samples_indices:
        curr_sample_value = train_dataset.samples[test_sample_index][attribute_index]
        curr_sample_class = train_dataset.sample_class[test_sample_index]
        if curr_sample_value in set_left_values:
            if curr_sample_class == left_class:
                num_correct_w_missing += 1
                num_correct_wo_missing += 1
        elif curr_sample_value in set_right_values:
            if curr_sample_class == right_class:
                num_correct_w_missing += 1
                num_correct_wo_missing += 1
        else:
            num_missing += 1
            if curr_sample_class == root_class:
                num_correct_w_missing += 1
    accuracy_w_missing = 100.0 * num_correct_w_missing / len(test_samples_indices)
    if num_missing != len(test_samples_indices):
        accuracy_wo_missing = (100.0 * num_correct_wo_missing
                               / (len(test_samples_indices) - num_missing))
    else:
        accuracy_wo_missing = None

    return accuracy_w_missing, accuracy_wo_missing, num_missing


def get_accuracy_root(train_dataset, root_class, test_samples_indices):
    num_correct = 0
    for test_sample_index in test_samples_indices:
        curr_sample_class = train_dataset.sample_class[test_sample_index]
        if curr_sample_class == root_class:
            num_correct += 1
    accuracy = 100.0 * num_correct / len(test_samples_indices)
    return accuracy, accuracy, 0


def save_fold_info(dataset_name, num_trials, criterion_name, num_samples, total_tests_done_ordered,
                   max_tests_done_ordered, min_tests_done_ordered, total_tests_per_attrib_ordered,
                   max_tests_per_attrib_ordered, min_tests_per_attrib_ordered,
                   total_tests_done_rev, max_tests_done_rev, min_tests_done_rev,
                   total_tests_per_attrib_rev, max_tests_per_attrib_rev, min_tests_per_attrib_rev,
                   total_tests_done_random_order, max_tests_done_random_order,
                   min_tests_done_random_order, total_tests_per_attrib_random_order,
                   max_tests_per_attrib_random_order, min_tests_per_attrib_random_order,
                   total_num_tests, max_num_tests, min_num_tests, total_num_fails_allowed,
                   max_num_fails_allowed, min_num_fails_allowed, total_num_valid_attributes,
                   max_num_valid_attributes, min_num_valid_attributes, total_theoretical_e_over_m,
                   max_theoretical_e_over_m, min_theoretical_e_over_m, U, L, p_L, output_split_char,
                   output_file_descriptor, sd_theoretical_e_over_m, sd_total_tests_done_ordered,
                   sd_total_tests_per_attrib_ordered, sd_total_tests_done_rev,
                   sd_total_tests_per_attrib_rev, sd_total_tests_done_random_order,
                   sd_total_tests_per_attrib_random_order, average_rank_ordered, min_rank_ordered,
                   max_rank_ordered, sd_rank_ordered, num_rank_found, average_time_criterion,
                   average_time_ordered, average_time_rev, average_time_random,
                   average_accuracy_best_according_to_criterion_w_missing_values,
                   average_accuracy_accepted_w_missing_values,
                   average_accuracy_best_according_to_criterion_wo_missing_values,
                   average_accuracy_accepted_wo_missing_values,
                   average_num_samples_missing_values_best_according_to_criterion,
                   average_num_samples_missing_values_accepted):
    line_list = [dataset_name,
                 str(num_trials),
                 str(U),
                 str(L),
                 str(1 - p_L),
                 str(p_L),
                 str(total_num_tests / num_trials),
                 str(max_num_tests),
                 str(min_num_tests),
                 str(total_num_fails_allowed / num_trials),
                 str(max_num_fails_allowed),
                 str(min_num_fails_allowed),
                 str(total_num_valid_attributes / num_trials),
                 str(max_num_valid_attributes),
                 str(min_num_valid_attributes),
                 str(total_theoretical_e_over_m / num_trials),
                 str(max_theoretical_e_over_m),
                 str(min_theoretical_e_over_m),
                 criterion_name,
                 str(num_samples),
                 str(total_tests_done_ordered / num_trials),
                 str(total_tests_per_attrib_ordered / num_trials),
                 str(max_tests_done_ordered),
                 str(max_tests_per_attrib_ordered),
                 str(min_tests_done_ordered),
                 str(min_tests_per_attrib_ordered),
                 str(total_tests_done_rev / num_trials),
                 str(total_tests_per_attrib_rev / num_trials),
                 str(max_tests_done_rev),
                 str(max_tests_per_attrib_rev),
                 str(min_tests_done_rev),
                 str(min_tests_per_attrib_rev),
                 str(total_tests_done_random_order / num_trials),
                 str(total_tests_per_attrib_random_order / num_trials),
                 str(max_tests_done_random_order),
                 str(max_tests_per_attrib_random_order),
                 str(min_tests_done_random_order),
                 str(min_tests_per_attrib_random_order),
                 str(datetime.datetime.now()),
                 str(sd_theoretical_e_over_m),
                 str(sd_total_tests_done_ordered),
                 str(sd_total_tests_per_attrib_ordered),
                 str(sd_total_tests_done_rev),
                 str(sd_total_tests_per_attrib_rev),
                 str(sd_total_tests_done_random_order),
                 str(sd_total_tests_per_attrib_random_order),
                 str(average_rank_ordered),
                 str(min_rank_ordered),
                 str(max_rank_ordered),
                 str(sd_rank_ordered),
                 str(num_rank_found),
                 str(average_time_criterion),
                 str(average_time_ordered),
                 str(average_time_rev),
                 str(average_time_random),
                 str(average_accuracy_best_according_to_criterion_w_missing_values),
                 str(average_accuracy_accepted_w_missing_values),
                 str(average_accuracy_best_according_to_criterion_wo_missing_values),
                 str(average_accuracy_accepted_wo_missing_values),
                 str(average_num_samples_missing_values_best_according_to_criterion),
                 str(average_num_samples_missing_values_accepted)]

    print(output_split_char.join(line_list), file=output_file_descriptor)


def main(dataset_names, datasets_filepaths, key_attrib_indices, class_attrib_indices, split_chars,
         missing_value_strings, num_samples, num_trials, U, L, p_L,
         output_csv_filepath, output_split_char=OUTPUT_SPLIT_CHAR):
    with open(output_csv_filepath, 'a') as fout:
        for dataset_number, filepath in enumerate(datasets_filepaths):
            if not os.path.exists(filepath) or not os.path.isfile(filepath):
                continue
            train_dataset = dataset.Dataset(filepath,
                                            key_attrib_indices[dataset_number],
                                            class_attrib_indices[dataset_number],
                                            split_chars[dataset_number],
                                            missing_value_strings[dataset_number])
            print('-'*100)
            print('Gini Twoing Monte Carlo')
            print()
            monte_carlo_tests_numbers(dataset_names[dataset_number],
                                      train_dataset,
                                      criteria.GiniTwoingMonteCarlo(),
                                      num_samples,
                                      num_trials,
                                      U,
                                      L,
                                      p_L,
                                      fout,
                                      output_split_char)
            fout.flush()
            print('-'*100)
            print('Fast Max Cut Chi Square Normalized')
            print()
            monte_carlo_tests_numbers(dataset_names[dataset_number],
                                      train_dataset,
                                      criteria.FastMaxCutChiSquareNormalized(),
                                      num_samples,
                                      num_trials,
                                      U,
                                      L,
                                      p_L,
                                      fout,
                                      output_split_char)
            fout.flush()
            # print('-'*100)
            # print('Fast Max Cut Chi Square Normalized P Value')
            # print()
            # monte_carlo_tests_numbers(dataset_names[dataset_number],
            #                           train_dataset,
            #                           criteria.FastMaxCutChiSquareNormalizedPValue(),
            #                           num_samples,
            #                           num_trials,
            #                           U,
            #                           L,
            #                           p_L,
            #                           fout,
            #                           output_split_char)
            # fout.flush()


if __name__ == '__main__':
    DATASET_NAMES = []
    DATASETS_FILEPATHS = []
    KEY_ATTRIB_INDICES = []
    CLASS_ATTRIB_INDICES = []
    SPLIT_CHARS = []
    MISSING_VALUE_STRINGS = []


    # Full datasets

    # Mushroom
    DATASET_NAMES.append('mushroom')
    DATASETS_FILEPATHS.append(os.path.join('.', 'datasets', 'mushroom', 'agaricus-lepiota.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(0)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # Adult census income
    DATASET_NAMES.append('adult census income')
    DATASETS_FILEPATHS.append(
        os.path.join('.', 'datasets', 'adult census income', 'adult_no_quotation_marks.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # KDD98 Target B
    KDD98_TRAIN_DB_FOLDER = os.path.join(
        '.', 'datasets', 'kdd98')

    DATASET_NAMES.append('kdd98_target_B')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_TRAIN_DB_FOLDER,
                                           'training_from_LRN.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-11)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # KDD98 Target D

    # 2 classes:
    KDD98_MULTICLASSES_TRAIN_DB_FOLDER = os.path.join(
        '.', 'datasets', 'kdd_multiclass', 'full')

    DATASET_NAMES.append('kdd98_multiclass_2')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_cup98LRN_multiclass_2.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # # 3 classes:
    DATASET_NAMES.append('kdd98_multiclass_3')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_cup98LRN_multiclass_3.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # 4 classes:
    DATASET_NAMES.append('kdd98_multiclass_4')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_cup98LRN_multiclass_4.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # 5 classes:
    DATASET_NAMES.append('kdd98_multiclass_5')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_cup98LRN_multiclass_5.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # 9 classes:
    DATASET_NAMES.append('kdd98_multiclass_9')
    DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
                                           'kdd98_cup98LRN_multiclass_9.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # # Connect4:
    # CONNECT_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Connect4-3Aggregat')
    # DATASET_NAMES.append('Connect4-3Aggregat')
    # DATASETS_FILEPATHS.append(os.path.join(CONNECT_TRAIN_DB_FOLDER,
    #                                        'Connect4-3Aggregat.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # Nursery-Agregate:
    NURSERY_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Nursery-Agregate')
    DATASET_NAMES.append('Nursery-Agregate_with_original')
    DATASETS_FILEPATHS.append(os.path.join(NURSERY_TRAIN_DB_FOLDER,
                                           'Nursery_original.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # CovTypeReduced:
    COVTYPE_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'CovTypeReduced')
    DATASET_NAMES.append('CovTypeReduced_with_original')
    DATASETS_FILEPATHS.append(os.path.join(COVTYPE_TRAIN_DB_FOLDER,
                                           'CovType_original_no_num.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # Car:
    CAR_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Car')
    DATASET_NAMES.append('Car_with_original')
    DATASETS_FILEPATHS.append(os.path.join(CAR_TRAIN_DB_FOLDER,
                                           'Car_orig_attributes.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # Contraceptive:
    NURSERY_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Contraceptive')
    DATASET_NAMES.append('Contraceptive_with_original')
    DATASETS_FILEPATHS.append(os.path.join(NURSERY_TRAIN_DB_FOLDER,
                                           'cmc_with_aggreg.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # Training datasets

    # # Mushroom
    # DATASET_NAMES.append('mushroom')
    # DATASETS_FILEPATHS.append(os.path.join('.', 'datasets', 'mushroom', 'training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(0)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # Adult census income
    # DATASET_NAMES.append('adult census income')
    # DATASETS_FILEPATHS.append(
    #     os.path.join('.', 'datasets', 'adult census income', 'training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # KDD98 Target D

    # # 2 classes:
    # KDD98_MULTICLASSES_TRAIN_DB_FOLDER = os.path.join(
    #     '.', 'datasets', 'kdd_multiclass')

    # DATASET_NAMES.append('kdd98_multiclass_2')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_2.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # 3 classes:
    # KDD98_MULTICLASSES_TRAIN_DB_FOLDER = os.path.join(
    #     '.', 'datasets', 'kdd_multiclass')

    # DATASET_NAMES.append('kdd98_multiclass_3')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_3.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # 4 classes:
    # DATASET_NAMES.append('kdd98_multiclass_4')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_4.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # 5 classes:
    # DATASET_NAMES.append('kdd98_multiclass_5')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_5.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # 9 classes:
    # DATASET_NAMES.append('kdd98_multiclass_9')
    # DATASETS_FILEPATHS.append(os.path.join(KDD98_MULTICLASSES_TRAIN_DB_FOLDER,
    #                                        'kdd98_from_LRN_multiclass_9.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # Connect4:
    # CONNECT_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Connect4-3Aggregat')
    # DATASET_NAMES.append('Connect4-3Aggregat')
    # DATASETS_FILEPATHS.append(os.path.join(CONNECT_TRAIN_DB_FOLDER,
    #                                        'Connect4-3Aggregat_training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # Nursery-Agregate:
    # NURSERY_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Nursery-Agregate')
    # DATASET_NAMES.append('Nursery-Agregate_with_original')
    # DATASETS_FILEPATHS.append(os.path.join(NURSERY_TRAIN_DB_FOLDER,
    #                                        'Nursery_original_training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # CovTypeReduced:
    # COVTYPE_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'CovTypeReduced')
    # DATASET_NAMES.append('CovTypeReduced')
    # DATASETS_FILEPATHS.append(os.path.join(COVTYPE_TRAIN_DB_FOLDER,
    #                                        'CovType_original_no_num_training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # Car:
    # CAR_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Car')
    # DATASET_NAMES.append('Car_with_original')
    # DATASETS_FILEPATHS.append(os.path.join(CAR_TRAIN_DB_FOLDER,
    #                                        'Car_training_orig_attributes.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    # # Contraceptive:
    # NURSERY_TRAIN_DB_FOLDER = os.path.join('.', 'datasets', 'Contraceptive')
    # DATASET_NAMES.append('Contraceptive_with_original')
    # DATASETS_FILEPATHS.append(os.path.join(NURSERY_TRAIN_DB_FOLDER,
    #                                        'cmc_with_aggreg_training.csv'))
    # KEY_ATTRIB_INDICES.append(None)
    # CLASS_ATTRIB_INDICES.append(-1)
    # SPLIT_CHARS.append(',')
    # MISSING_VALUE_STRINGS.append('?')

    OUTPUT_CSV_FILEPATH = os.path.join(
        '.',
        'outputs from datasets',
        'monte_carlo_tests_numbers_variable_num_samples_'
        'with_estimation_from_u_and_l_all_datasets_with_original_'
        'for_gini_twoing_and_fast_max_cut_chi_squared_normalized_'
        'tightest_over_50_trials_with_accuracies.csv')

    with open(OUTPUT_CSV_FILEPATH, 'a') as FOUT:
        FIELDS_LIST = ['Dataset', 'Number of Trials',
                       'U',
                       'L',
                       'p_U',
                       'p_L',
                       'Average Number of Tests (t)',
                       'Maximum Number of Tests (t)', 'Minimum Number of Tests (t)',
                       'Average Number of Failures (f - 1)', 'Maximum Number of Failures (f - 1)',
                       'Minimum Number of Failures (f - 1)',
                       'Average Number of Valid Attributes (m)',
                       'Maximum Number of Valid Attributes (m)',
                       'Minimum Number of Valid Attributes (m)',
                       'Average Theoretical Number of Tests per Attribute (E/m)',
                       'Maximum Theoretical Number of Tests per Attribute (E/m)',
                       'Minimum Theoretical Number of Tests per Attribute (E/m)',
                       'Criterion', 'Number of Samples',
                       'Average Number of Tests Needed - Ordered (E_ord)',
                       'Average Number of Tests Needed per Attribute - Ordered (E_ord/m)',
                       'Maximum Number of Tests Needed - Ordered (E_ord)',
                       'Maximum Number of Tests Needed per Attribute - Ordered (E_ord/m)',
                       'Minimum Number of Tests Needed - Ordered (E_ord)',
                       'Minimum Number of Tests Needed per Attribute - Ordered (E_ord/m)',
                       'Average Number of Tests Needed - Reversed (E_rev)',
                       'Average Number of Tests Needed per Attribute - Reversed (E_rev/m)',
                       'Maximum Number of Tests Needed - Reversed (E_rev)',
                       'Maximum Number of Tests Needed per Attribute - Reversed (E_rev/m)',
                       'Minimum Number of Tests Needed - Reversed (E_rev)',
                       'Minimum Number of Tests Needed per Attribute - Reversed (E_rev/m)',
                       'Average Number of Tests Needed - Random (E_random)',
                       'Average Number of Tests Needed per Attribute - Random (E_random/m)',
                       'Maximum Number of Tests Needed - Random (E_random)',
                       'Maximum Number of Tests Needed per Attribute - Random (E_random/m)',
                       'Minimum Number of Tests Needed - Random (E_random)',
                       'Minimum Number of Tests Needed per Attribute - Random (E_random/m)',
                       'Date Time',
                       'Standard Deviation of Theoretical Number of Tests per Attribute (sd(E/m))',
                       'Standard Deviation of Number of Tests Needed - Ordered (sd(E_ord))',
                       'Standard Deviation of Number of Tests Needed per Attribute'
                       ' - Ordered (sd(E_ord/m))',
                       'Standard Deviation of Number of Tests Needed - Reversed (sd(E_rev))',
                       'Standard Deviation of Number of Tests Needed per Attribute'
                       ' - Reversed (sd(E_rev/m))',
                       'Standard Deviation of Number of Tests Needed - Random (sd(E_random))',
                       'Standard Deviation of Number of Tests Needed per Attribute'
                       ' - Random (sd(E_random/m))',
                       'Average Rank of Accepted Attribute in Ordered',
                       'Min Rank of Accepted Attribute in Ordered',
                       'Max Rank of Accepted Attribute in Ordered',
                       'Standard Deviation of Rank of Accepted Attribute in Ordered',
                       'Number of Ordered Trials that Accepted an Attribute',
                       'Average Time taken to calculate the criterion on all attributes [s]',
                       'Average Time taken to do Monte Carlo on ordered attributes [s]',
                       'Average Time taken to do Monte Carlo on reversed attributes [s]',
                       'Average Time taken to do Monte Carlo on random attributes [s]',
                       'Average Accuracy on Best Attribute According to Criterion '
                       '(with missing values)',
                       'Average Accuracy on Accepted Attribute using Monte Carlo '
                       '(with missing values)',
                       'Average Accuracy on Best Attribute According to Criterion '
                       '(without missing values)',
                       'Average Accuracy on Accepted Attribute using Monte Carlo '
                       '(without missing values)',
                       'Average Number of Samples with Unkown Values for Best Attribute According'
                       ' to Criterion',
                       'Average Number of Samples with Unkown Values for Accepted Attribute'
                       ' using Monte Carlo']

        print(OUTPUT_SPLIT_CHAR.join(FIELDS_LIST), file=FOUT)
        FOUT.flush()

    # (U, L, p_L)
    PARAMETERS_LIST = [(0.4, 0.1, 0.95),
                       (0.4, 0.1, 0.99),
                       (0.3, 0.1, 0.95),
                       (0.3, 0.1, 0.99)]

    NUM_SAMPLES = [10, 30, 50, 100]
    NUM_TRIALS = 50
    for (curr_num_samples, (U, L, p_L)) in itertools.product(NUM_SAMPLES, PARAMETERS_LIST):
        main(DATASET_NAMES, DATASETS_FILEPATHS, KEY_ATTRIB_INDICES, CLASS_ATTRIB_INDICES,
             SPLIT_CHARS, MISSING_VALUE_STRINGS, curr_num_samples, NUM_TRIALS, U, L, p_L,
             OUTPUT_CSV_FILEPATH)
