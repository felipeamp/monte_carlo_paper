#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''This module calculates the number of tests/fails allowed in our Monte Carlo framework. It also
calculates the maximum expected value for the number of extra tests done by using it.
'''


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
    assert upper_p_value_threshold > lower_p_value_threshold
    assert upper_p_value_threshold <= 1.0 and upper_p_value_threshold > 0.0
    assert lower_p_value_threshold < 1.0 and lower_p_value_threshold >= 0.0
    assert prob_monte_carlo >= 0.0 and prob_monte_carlo <= 1.0

    return (0, 0)
    # return (num_tests, num_fails_allowed)

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
            or (num_fails_allowed < num_tests and num_tests > 0 and num_valid_nominal_attributes > 0))
    return 0.0
    # if num_tests == 0:
    #     return 0.0
    # return expected_total_num_tests
