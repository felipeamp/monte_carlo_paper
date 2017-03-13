#!/usr/bin/python3
# -*- coding: utf-8 -*-


'''This module calculates the number of tests/fails allowed in our Monte Carlo framework. It also
calculates the maximum expected value for the number of extra tests done by using it.
'''


from collections import defaultdict
import math


import numpy as np
from scipy.special import binom as binom_coef
from scipy.stats import binom as binom_stats


_MONTE_CARLO_EXPECTED_CACHE = defaultdict(list)
    # receives (t, f), returns a list where the m-th entry has the value of E when there are m valid
    # nominal attributes.
_MONTE_CARLO_T_F_CACHE = {} # receives (U, L, p, m), returns (t, f_allowed)


# TODO: conferir função abaixo se num_fails_allowed é isso mesmo ou é isso + 1.
def get_tests_and_fails_allowed(upper_p_value_threshold, lower_p_value_threshold, prob_monte_carlo,
                                num_valid_nominal_attributes):
    """Given the thresholds, probability confidence level and number of valid nominal attributes in
        the dataset, calculates how many tests should be done and how many fails are allowed to
        happen and still accept the attribute.

        Arguments:
            upper_p_value_threshold (float): the p-value-upper-threshold for our Monte Carlo
                framework. If an attribute has a p-value above this threshold, it will be rejected
                with probability `prob_monte_carlo`.
            lower_p_value_threshold (float): the p-value-lower-threshold for our Monte Carlo
                framework. If an attribute has a p-value below this threshold, it will be accepted
                with probability `prob_monte_carlo`.
            prob_monte_carlo (float): the probability of accepting an attribute with p-value smaller
                than `lower_p_value_threshold` and rejecting an attribute with p-value greater than
                `upper_p_value_threshold` for our Monte Carlo framework.
            num_valid_nominal_attributes (int): number of valid nominal attributes in the dataset.

        Returns:
            A tuple containing, in order:
                - number of tests;
                - maximum number of fails allowed.
        """
    def _is_ok_l(num_tests, num_fails_allowed, lower_p_value_threshold, prob_monte_carlo):
        return _get_s(1 - lower_p_value_threshold, num_tests, num_fails_allowed) >= prob_monte_carlo

    def _is_ok_u(num_tests, num_fails_allowed, upper_p_value_threshold, prob_monte_carlo,
                 num_valid_nominal_attributes):
        return (_get_s(1 - upper_p_value_threshold, num_tests, num_fails_allowed)
                <= 1. - math.pow(prob_monte_carlo, 1. / num_valid_nominal_attributes))


    assert upper_p_value_threshold > lower_p_value_threshold
    assert upper_p_value_threshold <= 1.0 and upper_p_value_threshold > 0.0
    assert lower_p_value_threshold < 1.0 and lower_p_value_threshold >= 0.0
    assert prob_monte_carlo >= 0.0 and prob_monte_carlo <= 1.0

    if (upper_p_value_threshold,
            lower_p_value_threshold,
            prob_monte_carlo,
            num_valid_nominal_attributes) in _MONTE_CARLO_T_F_CACHE:
        return _MONTE_CARLO_T_F_CACHE[(upper_p_value_threshold,
                                      lower_p_value_threshold,
                                      prob_monte_carlo,
                                      num_valid_nominal_attributes)]
    # We need to calculate these new values of num_tests and num_fails_allowed.
    num_fails_allowed = 1
    while True:
        # Let's get the largest t that satisfies the condition for L.
        # Note that, since s(x, t, num_fails_allowed) decreases as t increases, all t smaller than
        # the one we'll find satisty the condition for L.
        num_tests_low = num_fails_allowed + 1
        num_tests_high = 1000

        # # DEBUG:
        # print('-'*80)
        # print('num_fails_allowed =', num_fails_allowed)
        # print('num_tests_low =', num_tests_low)
        # print('num_tests_high =', num_tests_high)

        while num_tests_high > num_tests_low:
            num_tests_mid = (num_tests_low + num_tests_high + 1) // 2 # Round up
            if _is_ok_l(num_tests_mid, num_fails_allowed, lower_p_value_threshold, prob_monte_carlo):
                num_tests_low = num_tests_mid

                # # DEBUG:
                # print('num_tests_low =', num_tests_low)
                # print('num_tests_high =', num_tests_high)

            else:
                num_tests_high = num_tests_mid - 1

                # # DEBUG:
                # print('num_tests_low =', num_tests_low)
                # print('num_tests_high =', num_tests_high)

        if not _is_ok_l(num_tests_high,
                        num_fails_allowed,
                        lower_p_value_threshold,
                        prob_monte_carlo):
            continue
        else:
            largest_num_tests_l = num_tests_high

        # Let's get the smallest num_tests that satisfies the condition for U.
        # Note that, since s(x, num_tests, num_fails_allowed) decreases as num_tests increases, all
        # num_tests larger than the one we'll find satisty the condition for U.
        num_tests_low = num_fails_allowed + 1
        num_tests_high = 1000

        # # DEBUG:
        # print('-'*80)
        # print('num_fails_allowed =', num_fails_allowed)
        # print('num_tests_low =', num_tests_low)
        # print('num_tests_high =', num_tests_high)

        while num_tests_high > num_tests_low:
            num_tests_mid = (num_tests_low + num_tests_high) // 2
            if _is_ok_u(num_tests_mid,
                        num_fails_allowed,
                        upper_p_value_threshold,
                        prob_monte_carlo,
                        num_attrib):
                num_tests_high = num_tests_mid

                # # DEBUG:
                # print('num_tests_low =', num_tests_low)
                # print('num_tests_high =', num_tests_high)

            else:
                num_tests_low = num_tests_mid + 1

                # # DEBUG:
                # print('num_tests_low =', num_tests_low)
                # print('num_tests_high =', num_tests_high)

        if not _is_ok_u(num_tests_high, num_fails_allowed, upper_p_value_threshold, prob_monte_carlo, num_attrib):
            continue
        else:
            smallest_num_tests_u = num_tests_high

        if smallest_num_tests_u <= largest_num_tests_l:
            _MONTE_CARLO_T_F_CACHE[(upper_p_value_threshold,
                                   lower_p_value_threshold,
                                   prob_monte_carlo,
                                   num_valid_nominal_attributes)] = (smallest_num_tests_u,
                                                                     num_fails_allowed)
            return smallest_num_tests_u, num_fails_allowed
        num_fails_allowed += 1


def get_expected_total_num_tests(num_tests, num_fails_allowed, num_valid_nominal_attributes):
    """Given how many tests should be done and how many fails are allowed to happen and still accept
    an attribute, returns how many tests are expected to be done in this dataset, for the worst-case
    attributes p-value distributions.

    It is calculated using the stronger bound given in Theorem 2.

        Arguments:
            num_tests (int): number of tests to be done per attribute.
            num_fails_allowed (int): maximum number of fails an attribute can have and still be
                accepted.
            num_valid_nominal_attributes (int): number of valid nominal attributes in the dataset.

        Returns:
            The expected value of how many tests will be done in this dataset, for the attributes'
                worst-case p-value distribution.
        """
    assert (num_tests == 0
            or (num_fails_allowed < num_tests
                and num_tests > 0
                and num_valid_nominal_attributes > 0))

    if num_tests == 0:
        return 0.0
    if (num_tests, num_fails_allowed, num_valid_nominal_attributes) in _MONTE_CARLO_EXPECTED_CACHE:
        return _MONTE_CARLO_EXPECTED_CACHE[
            (num_tests, num_fails_allowed, num_valid_nominal_attributes)]

    # We need to calculate this new cost
    s_array = _get_s_coef_array(num_tests, num_fails_allowed)
    ds_array = _get_s_derivative_coef_array(num_tests, num_fails_allowed)
    max_value_for_root = _get_max_value_for_root(num_tests, num_fails_allowed)

    starting_m = len(_MONTE_CARLO_EXPECTED_CACHE[(num_tests, num_fails_allowed)]) + 1
    o_i = _MONTE_CARLO_EXPECTED_CACHE[(num_tests, num_fails_allowed)][-1]
    else:
        o_i = get_o_1(num_tests)#, num_fails_allowed, s_array, ds_array, max_value_for_root)
        print('o_1 =', o_i)
        save_t_f_m(num_tests, num_fails_allowed, 1, o_i)
        starting_m = 2

    for i in range(starting_m, m + 1):
        print('='*80)
        print('i =', i)
        r_i_prev = get_r_i(num_tests, num_fails_allowed, s_array, ds_array, o_i, max_value_for_root)
        # DEBUG
        print('r_{i-1}:')
        print(r_i_prev)
        print('*'*50)
        print('Calculating possible increments:')
        max_increment_found = max(get_increment_o_i(q, num_tests, num_fails_allowed, o_i)
                                  for q in r_i_prev)
        o_i += max_increment_found
        print('*'*50)
        print('max_increment_found =', max_increment_found)
        print('o_i =', o_i)
        save_t_f_m(num_tests, num_fails_allowed, i, o_i)

    # DEBUG
    print('='*80)
    print('num_tests =', num_tests)
    print('num_fails_allowed =', num_fails_allowed)
    print('m =', m)
    print('o_m =', o_i)
    print('E =', o_i)
    print('E/m =', o_i/m)
    print('='*80)

    return o_i


def _get_max_value_for_root(t, num_fails_allowed):
    return 1. - num_fails_allowed / t


def _get_s_derivative_coef_array(t, num_fails_allowed):
    ret = np.zeros(t + 1, dtype=float)
    for j in range(0, num_fails_allowed):
        curr_coef = 0.0
        for i in range(j, num_fails_allowed):
            if (i - j) & 1: # (i - j) is odd, thus (-1)**(i-j) == -1
                curr_coef -= _binom_coef(t, i) * _binom_coef(i, j)
            else: # (i - j) is even, thus (-1)**(i-j) == 1
                curr_coef += _binom_coef(t, i) * _binom_coef(i, j)
        ret[j + 1] = (t - j) * curr_coef
    return ret


def _get_s_coef_array(t, num_fails_allowed):
    ret = np.zeros(t + 1, dtype=float)
    for k in range(0, num_fails_allowed):
        curr_coef = 0.0
        for i in range(k, num_fails_allowed):
            if (i - k) & 1: # (i - k) is odd, thus (-1)**(i-k) == -1
                curr_coef -= _binom_coef(t, i) * _binom_coef(i, k)
            else: # (i - k) is even, thus (-1)**(i-k) == 1
                curr_coef += _binom_coef(t, i) * _binom_coef(i, k)
        ret[k] = curr_coef
    return ret


def get_r_i(t, f, s_array, ds_array, o_i, max_value_for_root):
    coef_array = np.zeros(2 + s_array.shape[0], dtype=float)

    coef_array[2:] += ds_array * (t - o_i - f) - s_array * f
    coef_array[1:-1] += ds_array * (2.0 * (o_i - t) + f)
    coef_array[:-2] += ds_array * (t - o_i)
    coef_array[-1] += f

    roots = _get_poly_roots_in_interval(coef_array, 0.0, max_value_for_root)
    roots.add(0.0)
    roots.add(max_value_for_root)

    # # DEBUG
    # if len(roots) > 2:
    #     print('==*=='*20)
    #     print('\t\tFound {} roots!'.format(len(roots) - 2))
    #     print('\t\tIncluding the interval boundaries, we have')
    #     print('\t\troots =', roots)
    #     print('==*=='*20)

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


def get_increment_o_i(q, t, num_fails_allowed, o_i):
    s = _get_s(q, t, num_fails_allowed)
    ret = num_fails_allowed / (1. - q) + s * (t - o_i - num_fails_allowed / (1. - q))

    # # DEBUG
    # print('-'*40)
    # print('q =', q)
    # print('t =', t)
    # print('num_fails_allowed =', num_fails_allowed)
    # print('o_{i-1} =', o_i)
    # print('s =', s)
    # print('increment_o_i =', ret)

    return ret


def _get_s(p, t, num_fails_allowed):
    return binom_stats.cdf(num_fails_allowed - 1, t, 1. - p)


