#!/usr/bin/python3
# -*- coding: utf-8 -*-


'''Module used to calculate table right before section 3.1.
'''

import itertools

import monte_carlo


M = [5, 20, 50]
L_U = [(0.1, 0.4), (0.1, 0.25), (0.01, 0.02)]
P_C = [0.95, 0.99]


def main(output_filepath):
    with open(output_filepath, 'w') as fout:
        print('m;L;U;p_c;t;f;E;E/m', file=fout)
        fout.flush()
        for m, prob_monte_carlo, (L, U)  in itertools.product(M, P_C, L_U):
            num_tests, num_fails_allowed = monte_carlo.get_tests_and_fails_allowed(
                U, L, prob_monte_carlo, m)
            expected_total_num_tests = monte_carlo.get_expected_total_num_tests(
                num_tests,
                num_fails_allowed,
                m)
            print('{};{};{};{};{};{};{};{}'.format(m,
                                                   L,
                                                   U,
                                                   prob_monte_carlo,
                                                   num_tests,
                                                   num_fails_allowed + 1,
                                                   expected_total_num_tests,
                                                   expected_total_num_tests / m),
                  file=fout)
            fout.flush()



if __name__ == '__main__':
    main('table.csv')
