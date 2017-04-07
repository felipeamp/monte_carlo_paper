#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to run tests on Monte Carlo methods using decision trees of a single level/node.
'''


import datetime
import itertools
import os
import random
import timeit

import criteria
import dataset
import decision_tree

import numpy as np


#: Initial seed used in `random` and `numpy.random` modules.
RANDOM_SEED = 65537
#: Character used to split cells in the output csv file.
OUTPUT_SPLIT_CHAR = ','
#: Maximum depth allowed for the tree to grow.
MAX_DEPTH = 1
#: Minimum number of samples allowed in a TreeNode and still allowing it to split during training.
MIN_NUM_SAMPLES_ALLOWED = 1


def monte_carlo_experiment(dataset_name, train_dataset, criterion, num_samples, num_trials,
                           use_chi_sq_test, max_p_value_chi_sq, use_monte_carlo,
                           upper_p_value_threshold, lower_p_value_threshold, prob_monte_carlo,
                           output_file_descriptor, output_split_char, seed=RANDOM_SEED):
    """Runs `num_trials` experiments, each one randomly selecting `num_samples` valid samples to use
    for training and testing the tree in the rest of the dataset. Saves the training and
    classification information in the `output_file_descriptor` file.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    training_samples_indices = list(range(train_dataset.num_samples))

    num_valid_attributes_list = []

    num_tests_list = []
    num_fails_allowed_list = []

    theoretical_e_over_m_list = []
    e_list = []
    e_over_m_list = []

    num_times_accepted = 0
    accepted_position_list = []

    total_time_taken_list = []
    time_taken_tree_list = []
    time_taken_prunning_list = []
    time_taken_num_tests_fails_list = []
    time_taken_expected_tests_list = []

    accuracy_with_missing_values_list = []
    accuracy_without_missing_values_list = []
    num_samples_missing_values_list = []
    num_nodes_pruned_list = []

    for _ in range(num_trials):
        random.shuffle(training_samples_indices)
        curr_training_samples_indices = training_samples_indices[:num_samples]
        curr_test_samples_indices = training_samples_indices[num_samples:]

        tree = decision_tree.DecisionTree(criterion=criterion,
                                          is_monte_carlo_criterion=use_monte_carlo,
                                          upper_p_value_threshold=upper_p_value_threshold,
                                          lower_p_value_threshold=lower_p_value_threshold,
                                          prob_monte_carlo=prob_monte_carlo)

        # First let's train the tree and save the training information
        start_time = timeit.default_timer()
        (time_taken_prunning,
         num_nodes_prunned) = tree.train(dataset=train_dataset,
                                         training_samples_indices=curr_training_samples_indices,
                                         max_depth=MAX_DEPTH,
                                         min_samples_per_node=MIN_NUM_SAMPLES_ALLOWED,
                                         use_stop_conditions=use_chi_sq_test,
                                         max_p_value_chi_sq=max_p_value_chi_sq,
                                         calculate_expected_tests=use_monte_carlo)
        total_time_taken = timeit.default_timer() - start_time

        while (sorted(tree.get_root_node().class_index_num_samples)[-2] == 0
               or sum(tree.get_root_node().valid_nominal_attribute) == 0):
            random.shuffle(training_samples_indices)
            curr_training_samples_indices = training_samples_indices[:num_samples]
            curr_test_samples_indices = training_samples_indices[num_samples: 2 * num_samples]

            start_time = timeit.default_timer()
            (time_taken_prunning,
             num_nodes_prunned) = tree.train(dataset=train_dataset,
                                             training_samples_indices=curr_training_samples_indices,
                                             max_depth=MAX_DEPTH,
                                             min_samples_per_node=MIN_NUM_SAMPLES_ALLOWED,
                                             use_stop_conditions=use_chi_sq_test,
                                             max_p_value_chi_sq=max_p_value_chi_sq,
                                             calculate_expected_tests=use_monte_carlo)
            total_time_taken = timeit.default_timer() - start_time

        num_valid_attributes_list.append(sum(tree.get_root_node().valid_nominal_attribute))

        root_node = tree.get_root_node()

        num_tests_list.append(root_node.num_tests)
        num_fails_allowed_list.append(root_node.num_fails_allowed)
        theoretical_e_over_m_list.append(
            root_node.total_expected_num_tests / num_valid_attributes_list[-1])

        if root_node.node_split is not None:
            num_times_accepted += 1
            e_list.append(root_node.node_split.total_num_tests_needed)
            e_over_m_list.append(
                root_node.node_split.total_num_tests_needed / num_valid_attributes_list[-1])
            accepted_position_list.append(root_node.node_split.accepted_position)

        total_time_taken_list.append(total_time_taken)
        time_taken_prunning_list.append(time_taken_prunning)
        time_taken_num_tests_fails_list.append(root_node.get_subtree_time_num_tests_fails())
        time_taken_expected_tests_list.append(root_node.get_subtree_time_expected_tests())
        time_taken_tree_list = (total_time_taken
                                - time_taken_prunning_list[-1]
                                - time_taken_num_tests_fails_list[-1]
                                - time_taken_expected_tests_list[-1])

        num_nodes_pruned_list.append(num_nodes_prunned)

        # Time to test this tree's classification and save the classification information
        (_,
         num_correct_classifications_w_unkown,
         num_correct_classifications_wo_unkown,
         _,
         _,
         _,
         num_unkown,
         _) = tree.test(curr_test_samples_indices)

        accuracy_with_missing_values_list.append(
            num_correct_classifications_w_unkown / len(curr_test_samples_indices))
        accuracy_without_missing_values_list.append(
            num_correct_classifications_wo_unkown / len(curr_test_samples_indices))
        num_samples_missing_values_list.append(num_unkown)

    save_fold_info(dataset_name, num_samples, num_trials, criterion.name, use_chi_sq_test,
                   max_p_value_chi_sq, use_monte_carlo, criteria.ORDER_RANDOMLY,
                   upper_p_value_threshold, lower_p_value_threshold, prob_monte_carlo,
                   np.array(num_tests_list), np.array(num_fails_allowed_list),
                   np.array(num_valid_attributes_list), np.array(theoretical_e_over_m_list),
                   np.array(e_list), np.array(e_over_m_list), np.array(accepted_position_list),
                   num_times_accepted, np.array(total_time_taken_list),
                   np.array(time_taken_tree_list), np.array(time_taken_prunning_list),
                   np.array(time_taken_num_tests_fails_list),
                   np.array(time_taken_expected_tests_list),
                   np.array(accuracy_with_missing_values_list),
                   np.array(accuracy_without_missing_values_list),
                   np.array(num_samples_missing_values_list), np.array(num_nodes_pruned_list),
                   output_split_char, output_file_descriptor)


def save_fold_info(dataset_name, num_samples, num_trials, criterion_name, use_chi_sq_test,
                   max_p_value_chi_sq, use_monte_carlo, is_random_ordering, upper_p_value_threshold,
                   lower_p_value_threshold, prob_monte_carlo, num_tests_array,
                   num_fails_allowed_array, num_valid_attributes_array, theoretical_e_over_m_array,
                   e_array, e_over_m_array, accepted_position_array, num_times_accepted,
                   total_time_taken_array, time_taken_tree_array, time_taken_prunning_array,
                   time_taken_num_tests_fails_array, time_taken_expected_tests_array,
                   accuracy_with_missing_values_array, accuracy_without_missing_values_array,
                   num_samples_missing_values_array, num_nodes_pruned_array, output_split_char,
                   output_file_descriptor):
    """Saves the experiment information in the CSV file.
    """
    assert num_trials > 0
    line_list = [str(datetime.datetime.now()),
                 dataset_name,
                 str(num_samples),
                 str(num_trials),
                 criterion_name,
                 str(MIN_NUM_SAMPLES_ALLOWED),

                 str(use_chi_sq_test),
                 str(max_p_value_chi_sq),

                 str(use_monte_carlo),
                 str(is_random_ordering),
                 str(upper_p_value_threshold),
                 str(lower_p_value_threshold),
                 str(prob_monte_carlo),

                 str(np.mean(num_tests_array)),
                 str(np.amax(num_tests_array)),
                 str(np.amin(num_tests_array)),

                 str(np.mean(num_fails_allowed_array)),
                 str(np.amax(num_fails_allowed_array)),
                 str(np.amin(num_fails_allowed_array)),

                 str(np.mean(num_valid_attributes_array)),
                 str(np.amax(num_valid_attributes_array)),
                 str(np.amin(num_valid_attributes_array)),

                 str(np.mean(theoretical_e_over_m_array)),
                 str(np.amax(theoretical_e_over_m_array)),
                 str(np.amin(theoretical_e_over_m_array)),
                 str(np.std(theoretical_e_over_m_array)),

                 str(np.mean(e_array)),
                 str(np.amax(e_array)),
                 str(np.amin(e_array)),
                 str(np.std(e_array)),

                 str(np.mean(e_over_m_array)),
                 str(np.amax(e_over_m_array)),
                 str(np.amin(e_over_m_array)),
                 str(np.std(e_over_m_array)),

                 str(num_times_accepted),
                 str(np.mean(accepted_position_array)),
                 str(np.amax(accepted_position_array)),
                 str(np.amin(accepted_position_array)),
                 str(np.std(accepted_position_array)),

                 str(np.mean(total_time_taken_array)),
                 str(np.mean(time_taken_tree_array)),
                 str(np.mean(time_taken_prunning_array)),
                 str(np.mean(time_taken_num_tests_fails_array)),
                 str(np.mean(time_taken_expected_tests_array)),

                 str(np.mean(accuracy_with_missing_values_array)),
                 str(np.mean(accuracy_without_missing_values_array)),
                 str(np.mean(num_samples_missing_values_array)),

                 str(np.mean(num_nodes_pruned_array))]

    print(output_split_char.join(line_list), file=output_file_descriptor)


def main(dataset_names, datasets_filepaths, key_attrib_indices, class_attrib_indices, split_chars,
         missing_value_strings, num_samples, num_trials, use_chi_sq_test, max_p_value_chi_sq,
         use_monte_carlo, use_random_ordering, upper_p_value_threshold, lower_p_value_threshold,
         prob_monte_carlo, output_csv_filepath, output_split_char=OUTPUT_SPLIT_CHAR):
    with open(output_csv_filepath, 'a') as fout:
        for dataset_number, filepath in enumerate(datasets_filepaths):
            if not os.path.exists(filepath) or not os.path.isfile(filepath):
                continue

            train_dataset = dataset.Dataset(filepath,
                                            key_attrib_indices[dataset_number],
                                            class_attrib_indices[dataset_number],
                                            split_chars[dataset_number],
                                            missing_value_strings[dataset_number])
            criteria.ORDER_RANDOMLY = use_random_ordering

            print('-'*100)
            print('Gini Gain')
            print()
            monte_carlo_experiment(dataset_names[dataset_number],
                                   train_dataset,
                                   criteria.GiniGain(),
                                   num_samples,
                                   num_trials,
                                   use_chi_sq_test,
                                   max_p_value_chi_sq,
                                   use_monte_carlo,
                                   upper_p_value_threshold,
                                   lower_p_value_threshold,
                                   prob_monte_carlo,
                                   fout,
                                   output_split_char)
            fout.flush()
            print('-'*100)
            print('Twoing')
            print()
            monte_carlo_experiment(dataset_names[dataset_number],
                                   train_dataset,
                                   criteria.GiniGain(),
                                   num_samples,
                                   num_trials,
                                   use_chi_sq_test,
                                   max_p_value_chi_sq,
                                   use_monte_carlo,
                                   upper_p_value_threshold,
                                   lower_p_value_threshold,
                                   prob_monte_carlo,
                                   fout,
                                   output_split_char)
            fout.flush()
            print('-'*100)
            print('Gain Ratio')
            print()
            monte_carlo_experiment(dataset_names[dataset_number],
                                   train_dataset,
                                   criteria.GiniGain(),
                                   num_samples,
                                   num_trials,
                                   use_chi_sq_test,
                                   max_p_value_chi_sq,
                                   use_monte_carlo,
                                   upper_p_value_threshold,
                                   lower_p_value_threshold,
                                   prob_monte_carlo,
                                   fout,
                                   output_split_char)
            fout.flush()


if __name__ == '__main__':
    DATASET_NAMES = []
    DATASETS_FILEPATHS = []
    KEY_ATTRIB_INDICES = []
    CLASS_ATTRIB_INDICES = []
    SPLIT_CHARS = []
    MISSING_VALUE_STRINGS = []

    DATASET_BASE_PATH = os.path.join('.', 'datasets')

    # Adult census income
    DATASET_NAMES.append('Adult Census Income')
    DATASETS_FILEPATHS.append(os.path.join(DATASET_BASE_PATH,
                                           'adult census income',
                                           'adult_no_quotation_marks.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('')

    # Car:
    DATASET_NAMES.append('Cars')
    DATASETS_FILEPATHS.append(os.path.join(DATASET_BASE_PATH,
                                           'cars'
                                           'cars.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append(None)

    # Mushroom
    DATASET_NAMES.append('Mushroom')
    DATASETS_FILEPATHS.append(os.path.join(DATASET_BASE_PATH,
                                           'mushroom',
                                           'agaricus-lepiota.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(0)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append('?')

    # Nursery:
    DATASET_NAMES.append('Nursery')
    DATASETS_FILEPATHS.append(os.path.join(DATASET_BASE_PATH,
                                           'nursery',
                                           'nursery.txt'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append(None)

    # Titanic Survive:
    DATASET_NAMES.append('Titanic Survive')
    DATASETS_FILEPATHS.append(os.path.join(DATASET_BASE_PATH,
                                           'titanic',
                                           'titanicSurvive.csv'))
    KEY_ATTRIB_INDICES.append(None)
    CLASS_ATTRIB_INDICES.append(-1)
    SPLIT_CHARS.append(',')
    MISSING_VALUE_STRINGS.append(None)


    OUTPUT_CSV_FILEPATH = os.path.join(
        '.',
        'outputs',
        'single_level_experiment_1.csv')

    with open(OUTPUT_CSV_FILEPATH, 'a') as FOUT:
        FIELDS_LIST = ['Date Time',
                       'Dataset',
                       'Number of Samples',
                       'Number of Trials',
                       'Criterion',
                       'Number of Samples to Force a Leaf',

                       'Uses Chi-Square Test',
                       'Maximum p-value Allowed by Chi-Square Test [between 0 and 1]',

                       'Uses Monte Carlo',
                       'Are Attributes in Random Order?',
                       'U [between 0 and 1]',
                       'L [between 0 and 1]',
                       'prob_monte_carlo [between 0 and 1]',

                       'Average Number of Tests (t)',
                       'Maximum Number of Tests (t)',
                       'Minimum Number of Tests (t)',

                       'Average Number of Fails Allowed (f - 1)',
                       'Maximum Number of Fails Allowed (f - 1)',
                       'Minimum Number of Fails Allowed (f - 1)',

                       'Average Number of Valid Attributes (m)',
                       'Maximum Number of Valid Attributes (m)',
                       'Minimum Number of Valid Attributes (m)',

                       'Average Theoretical Number of Tests per Attribute (E/m)',
                       'Maximum Theoretical Number of Tests per Attribute (E/m)',
                       'Minimum Theoretical Number of Tests per Attribute (E/m)',
                       'Standard Deviation of Theoretical Number of Tests per Attribute (sd(E/m))',

                       'Average Number of Tests Needed (E)',
                       'Maximum Number of Tests Needed (E)',
                       'Minimum Number of Tests Needed (E)',
                       'Standard Deviation of Number of Tests Needed (sd(E))',

                       'Average Number of Tests Needed per Attribute(E/m)',
                       'Maximum Number of Tests Needed per Attribute (E/m)',
                       'Minimum Number of Tests Needed per Attribute (E/m)',
                       'Standard Deviation of Number of Tests Needed per Attribute (sd(E/m))',

                       'Number of Experiments with Accepted Attributes',
                       'Average Position of Accepted Attribute',
                       'Min Position of Accepted Attribute',
                       'Max Position of Accepted Attribute',
                       'Standard Deviation of Position of Accepted Attribute',

                       'Average Total Time Taken [s]',
                       'Average Time Taken to Create Tree [s]',
                       'Average Time Taken Prunning Trivial Subtrees [s]',
                       'Average Time Taken to Calculate t and f [s]',
                       'Average Time Taken to Calculate E [s]',

                       'Average Accuracy Percentage (with missing values)',
                       'Average Accuracy Percentage (without missing values)',
                       'Average Number of Samples with Unkown Values for Accepted Attribute',

                       'Average Number of Nodes Pruned']

        print(OUTPUT_SPLIT_CHAR.join(FIELDS_LIST), file=FOUT)
        FOUT.flush()

    # (upper_p_value_threshold, lower_p_value_threshold, prob_monte_carlo)
    PARAMETERS_LIST = [(0.4, 0.1, 0.95),
                       (0.4, 0.1, 0.99),
                       (0.3, 0.1, 0.95),
                       (0.3, 0.1, 0.99)]

    NUM_SAMPLES = [10, 30, 50, 100]
    NUM_TRIALS = 50

    USE_RANDOM = [False, True]
    MAX_P_VALUE_CHI_SQ = [0.1]

    for curr_num_samples in NUM_SAMPLES:
        # Run without any bias treatment
        main(DATASET_NAMES,
             DATASETS_FILEPATHS,
             KEY_ATTRIB_INDICES,
             CLASS_ATTRIB_INDICES,
             SPLIT_CHARS,
             MISSING_VALUE_STRINGS,
             curr_num_samples,
             NUM_TRIALS,
             use_chi_sq_test=False,
             max_p_value_chi_sq=None,
             use_monte_carlo=False,
             use_random_ordering=False,
             upper_p_value_threshold=None,
             lower_p_value_threshold=None,
             prob_monte_carlo=None,
             output_csv_filepath=OUTPUT_CSV_FILEPATH)
        # Run with Chi Square test
        for curr_max_p_value_chi_sq in MAX_P_VALUE_CHI_SQ:
            main(DATASET_NAMES,
                 DATASETS_FILEPATHS,
                 KEY_ATTRIB_INDICES,
                 CLASS_ATTRIB_INDICES,
                 SPLIT_CHARS,
                 MISSING_VALUE_STRINGS,
                 curr_num_samples,
                 NUM_TRIALS,
                 use_chi_sq_test=True,
                 max_p_value_chi_sq=curr_max_p_value_chi_sq,
                 use_monte_carlo=False,
                 use_random_ordering=False,
                 upper_p_value_threshold=None,
                 lower_p_value_threshold=None,
                 prob_monte_carlo=None,
                 output_csv_filepath=OUTPUT_CSV_FILEPATH)
        # Run with Monte Carlo Framework
        for (curr_use_random_ordering,
             (curr_upper_p_value_threshold,
              curr_lower_p_value_threshold,
              curr_prob_monte_carlo)) in itertools.product(USE_RANDOM, PARAMETERS_LIST):
            main(DATASET_NAMES,
                 DATASETS_FILEPATHS,
                 KEY_ATTRIB_INDICES,
                 CLASS_ATTRIB_INDICES,
                 SPLIT_CHARS,
                 MISSING_VALUE_STRINGS,
                 curr_num_samples,
                 NUM_TRIALS,
                 use_chi_sq_test=False,
                 max_p_value_chi_sq=None,
                 use_monte_carlo=True,
                 use_random_ordering=curr_use_random_ordering,
                 upper_p_value_threshold=curr_upper_p_value_threshold,
                 lower_p_value_threshold=curr_lower_p_value_threshold,
                 prob_monte_carlo=curr_prob_monte_carlo,
                 output_csv_filepath=OUTPUT_CSV_FILEPATH)
