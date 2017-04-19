#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""This module calculates the number of tests/fails allowed in our Monte Carlo framework. It also
calculates the maximum expected value for the number of extra tests done by using it.
"""


from collections import defaultdict
import math


import numpy as np
from scipy.special import binom as binom_coef
from scipy.stats import binom as binom_stats

#: Cache where, given (upper_p_value_threshold, lower_p_value_threshold, prob_monte_carlo,
#: num_valid_attributes), returns returns (num_tests, num_fails_allowed).
_MONTE_CARLO_T_F_CACHE = {}
#: Cache where, given (num_tests, num_fails_allowed), returns a list where the m-th entry has the
# value of expected number of total tests when there are m valid nominal attributes.
_MONTE_CARLO_EXPECTED_CACHE = defaultdict(list)


def clean_caches():
    """Cleans `_MONTE_CARLO_T_F_CACHE` and `_MONTE_CARLO_EXPECTED_CACHE`.
    """
    global _MONTE_CARLO_T_F_CACHE
    _MONTE_CARLO_T_F_CACHE = {}
    global _MONTE_CARLO_EXPECTED_CACHE
    _MONTE_CARLO_EXPECTED_CACHE = defaultdict(list)


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

    def _find_largest_num_tests_l(num_fails_allowed, lower_p_value_threshold, prob_monte_carlo):
        """Returns the largest num_tests that satisfies the condition for L using
        exponencial search. Note that, since s(x, num_tests, num_fails_allowed) decreases
        as num_tests increases, every num_tests smaller than the one we'll find also satisfies
        the condition for L.
        """
        num_tests_ini = num_fails_allowed + 1
        if not _is_ok_l(num_tests_ini,
                        num_fails_allowed,
                        lower_p_value_threshold,
                        prob_monte_carlo):
            # No value for num_tests will satisfy the inequality for lower_p_value_threshold for
            # this value of num_fails_allowed.
            return None

        # Exponential Search
        num_exp = 0
        num_tests_high = num_tests_ini + 1
        num_tests_low = num_tests_ini
        while _is_ok_l(num_tests_high,
                       num_fails_allowed,
                       lower_p_value_threshold,
                       prob_monte_carlo):
            num_tests_low = num_tests_high
            num_exp += 1
            num_tests_high += 2**num_exp

        # Binary Search
        while num_tests_high > num_tests_low:
            # Loop Invariant:
            # At least one of the bounds (low or high) will satisfy the inequality for
            # lower_p_value_threshold.

            # Base case:
            # When num_tests_high == num_tests_low + 1, the midpoint will be num_tests_high. If it
            # satisfies the inequality, that's our answer. Otherwise, it is num_tests_low.
            num_tests_mid = (num_tests_low + num_tests_high + 1) // 2 # Round up
            if _is_ok_l(num_tests_mid,
                        num_fails_allowed,
                        lower_p_value_threshold,
                        prob_monte_carlo):
                num_tests_low = num_tests_mid
            else:
                num_tests_high = num_tests_mid - 1

        return num_tests_high

    def _find_smallest_num_tests_u(num_fails_allowed, upper_p_value_threshold, prob_monte_carlo):
        """Returns the smallest num_tests that satisfies the condition for U using
        exponential search. Note that, since s(x, num_tests, num_fails_allowed) decreases
        as num_tests increases, every num_tests larger than the one we'll find also satisfies
        the condition for U.
        """
        num_tests_ini = num_fails_allowed + 1
        if _is_ok_u(num_tests_ini,
                    num_fails_allowed,
                    upper_p_value_threshold,
                    prob_monte_carlo,
                    num_valid_nominal_attributes):
            # If num_tests_ini satisfies the inequality, that's our answer.
            return num_tests_ini

        # Exponential Search
        num_exp = 0
        num_tests_high = num_tests_ini + 1
        num_tests_low = num_tests_ini
        while not _is_ok_u(num_tests_high,
                           num_fails_allowed,
                           upper_p_value_threshold,
                           prob_monte_carlo,
                           num_valid_nominal_attributes):
            num_tests_low = num_tests_high
            num_exp += 1
            num_tests_high += 2**num_exp

        # Binary Search
        while num_tests_high > num_tests_low:
            # Loop Invariant:
            # At least one of the bounds (low or high) will satisfy the inequality for
            # upper_p_value_threshold.

            # Base case:
            # When num_tests_high == num_tests_low + 1, the midpoint will be num_tests_low. If it
            # satisfies the inequality, that's our answer. Otherwise, it is num_tests_high.
            num_tests_mid = (num_tests_low + num_tests_high) // 2 # Round down.
            if _is_ok_u(num_tests_mid,
                        num_fails_allowed,
                        upper_p_value_threshold,
                        prob_monte_carlo,
                        num_valid_nominal_attributes):
                num_tests_high = num_tests_mid
            else:
                num_tests_low = num_tests_mid + 1

        return num_tests_high


    assert upper_p_value_threshold > lower_p_value_threshold
    assert upper_p_value_threshold <= 1.0 and upper_p_value_threshold > 0.0
    assert lower_p_value_threshold < 1.0 and lower_p_value_threshold >= 0.0
    assert prob_monte_carlo >= 0.0 and prob_monte_carlo <= 1.0
    assert num_valid_nominal_attributes > 0

    # Cache Lookup
    if (upper_p_value_threshold,
            lower_p_value_threshold,
            prob_monte_carlo,
            num_valid_nominal_attributes) in _MONTE_CARLO_T_F_CACHE:
        return _MONTE_CARLO_T_F_CACHE[(upper_p_value_threshold,
                                       lower_p_value_threshold,
                                       prob_monte_carlo,
                                       num_valid_nominal_attributes)]

    # We need to calculate these new values of num_tests and num_fails_allowed.
    num_fails_allowed = 0
    while True:
        largest_num_tests_l = _find_largest_num_tests_l(num_fails_allowed,
                                                        lower_p_value_threshold,
                                                        prob_monte_carlo)
        if largest_num_tests_l is None:
            num_fails_allowed += 1
            continue

        smallest_num_tests_u = _find_smallest_num_tests_u(num_fails_allowed,
                                                          upper_p_value_threshold,
                                                          prob_monte_carlo)
        if smallest_num_tests_u is None:
            num_fails_allowed += 1
            continue

        if smallest_num_tests_u <= largest_num_tests_l:
            # Found the answer! Save in cache and return it.
            _MONTE_CARLO_T_F_CACHE[(upper_p_value_threshold,
                                    lower_p_value_threshold,
                                    prob_monte_carlo,
                                    num_valid_nominal_attributes)] = (smallest_num_tests_u,
                                                                      num_fails_allowed)
            return (smallest_num_tests_u, num_fails_allowed)
        num_fails_allowed += 1

    return (None, None)


def get_expected_total_num_tests(num_tests, num_fails_allowed, num_valid_nominal_attributes):
    """Given how many tests should be done and how many fails are allowed to happen and still accept
    an attribute, returns how many tests are expected to be done in this dataset, for the worst-case
    attributes p-value distributions.

    It is calculated using the stronger bound given in Theorem 2.

        Args:
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
    if (num_valid_nominal_attributes
            < len(_MONTE_CARLO_EXPECTED_CACHE[(num_tests, num_fails_allowed)])):
        return _MONTE_CARLO_EXPECTED_CACHE[(num_tests, num_fails_allowed)][
            num_valid_nominal_attributes - 1]

    # We need to calculate this new cost
    s_array = _get_s_coef_array(num_tests, num_fails_allowed)
    ds_array = _get_s_derivative_coef_array(num_tests, num_fails_allowed)
    max_value_for_root = _get_max_value_for_root(num_tests, num_fails_allowed)

    starting_num_attributes = len(_MONTE_CARLO_EXPECTED_CACHE[(num_tests, num_fails_allowed)]) + 1
    if starting_num_attributes == 1:
        o_i = num_tests
        _MONTE_CARLO_EXPECTED_CACHE[(num_tests, num_fails_allowed)].append(o_i)
    else:
        o_i = _MONTE_CARLO_EXPECTED_CACHE[(num_tests, num_fails_allowed)][-1]


    for _ in range(starting_num_attributes + 1, num_valid_nominal_attributes + 1):
        r_i_prev = _get_r_i(num_tests,
                            num_fails_allowed,
                            s_array,
                            ds_array,
                            o_i,
                            max_value_for_root)
        max_increment_found = max(_get_increment_o_i(root, num_tests, num_fails_allowed, o_i)
                                  for root in r_i_prev)
        o_i += max_increment_found
        _MONTE_CARLO_EXPECTED_CACHE[(num_tests, num_fails_allowed)].append(o_i)

    return o_i


def _get_max_value_for_root(num_tests, num_fails_allowed):
    return 1. - (num_fails_allowed + 1) / num_tests


def _get_s_derivative_coef_array(num_tests, num_fails_allowed):
    ret = np.zeros(num_tests + 1, dtype=float)
    for j in range(num_fails_allowed + 1):
        curr_coef = 0.0
        for i in range(j, num_fails_allowed + 1):
            if (i - j) & 1: # (i - j) is odd, thus (-1)**(i-j) == -1
                curr_coef -= binom_coef(num_tests, i) * binom_coef(i, j)
            else: # (i - j) is even, thus (-1)**(i-j) == 1
                curr_coef += binom_coef(num_tests, i) * binom_coef(i, j)
        ret[j + 1] = (num_tests - j) * curr_coef
    return ret


def _get_s_coef_array(num_tests, num_fails_allowed):
    ret = np.zeros(num_tests + 1, dtype=float)
    for k in range(num_fails_allowed + 1):
        curr_coef = 0.0
        for i in range(k, num_fails_allowed + 1):
            if (i - k) & 1: # (i - k) is odd, thus (-1)**(i-k) == -1
                curr_coef -= binom_coef(num_tests, i) * binom_coef(i, k)
            else: # (i - k) is even, thus (-1)**(i-k) == 1
                curr_coef += binom_coef(num_tests, i) * binom_coef(i, k)
        ret[k] = curr_coef
    return ret


def _get_r_i(num_tests, num_fails_allowed, s_array, ds_array, o_i, max_value_for_root):
    coef_array = np.zeros(2 + s_array.shape[0], dtype=float)

    coef_array[2:] += (ds_array * (num_tests - o_i - num_fails_allowed - 1)
                       - s_array * (num_fails_allowed + 1))
    coef_array[1:-1] += ds_array * (2.0 * (o_i - num_tests) + num_fails_allowed + 1)
    coef_array[:-2] += ds_array * (num_tests - o_i)
    coef_array[-1] += num_fails_allowed + 1

    roots = _get_poly_roots_in_interval(coef_array, 0.0, max_value_for_root)
    roots.add(0.0)
    roots.add(max_value_for_root)

    return roots


def _get_poly_roots_in_interval(coef_array, min_value_for_root, max_value_for_root):
    roots = np.roots(coef_array)
    ret = set()
    for is_real, root in zip(np.isreal(roots), roots):
        if is_real and root.real >= min_value_for_root and root.real <= max_value_for_root:
            ret.add(float(root.real))
    return ret


def _get_increment_o_i(root, num_tests, num_fails_allowed, o_i):
    s = _get_s(root, num_tests, num_fails_allowed)
    ret = ((num_fails_allowed + 1) / (1. - root)
           + s * (num_tests - o_i - num_fails_allowed - 1) / (1. - root))
    return ret


def _get_s(prob_success, num_tests, num_fails_allowed):
    return binom_stats.cdf(num_fails_allowed, num_tests, 1. - prob_success)
