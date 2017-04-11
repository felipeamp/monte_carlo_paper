#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to run tests on Monte Carlo methods using decision trees of a single level/node.
'''


import datetime
import itertools
import os
import timeit

import criteria
import dataset
import decision_tree

import numpy as np


#: Initial seed used in `random` and `numpy.random` modules.
RANDOM_SEED = 65537
#: Character used to split cells in the output csv file.
OUTPUT_SPLIT_CHAR = ','
#: Number of folds to be used during cross-validation.
NUM_FOLDS = 10
#: Number of experiments to be done with each parameters combination.
NUM_TRIALS = 5


def monte_carlo_experiment(dataset_name, train_dataset, criterion, min_num_samples_allowed,
                           max_depth, num_trials, use_chi_sq_test, max_p_value_chi_sq,
                           use_monte_carlo, upper_p_value_threshold, lower_p_value_threshold,
                           prob_monte_carlo, output_file_descriptor, output_split_char,
                           seed=RANDOM_SEED):
    """Runs `num_trials` experiments, each one doing a stratified cross-validation in `NUM_FOLDS`
    folds. Saves the training and classification information in the `output_file_descriptor` file.
    """
    num_valid_attributes_list = []

    num_nodes_list = []
    tree_depth_list = []

    total_time_taken_list = []

    accuracy_with_missing_values_list = []
    accuracy_without_missing_values_list = []
    num_unkown_list = []
    num_nodes_pruned_list = []

    for trial_number in range(num_trials):
        print('*'*80)
        print('STARTING TRIAL #{}'.format(trial_number + 1))
        print()

        tree = decision_tree.DecisionTree(criterion=criterion,
                                          is_monte_carlo_criterion=use_monte_carlo,
                                          upper_p_value_threshold=upper_p_value_threshold,
                                          lower_p_value_threshold=lower_p_value_threshold,
                                          prob_monte_carlo=prob_monte_carlo)

        start_time = timeit.default_timer()
        (_,
         num_correct_classifications_w_unkown,
         num_correct_classifications_wo_unkown,
         _,
         _,
         _,
         num_unkown,
         _,
         _,
         num_nodes_prunned_per_fold,
         max_depth_per_fold,
         num_nodes_per_fold,
         num_valid_attributes_in_root) = tree.cross_validate(
             dataset=train_dataset,
             num_folds=NUM_FOLDS,
             max_depth=max_depth,
             min_samples_per_node=min_num_samples_allowed,
             is_stratified=True,
             print_tree=False,
             seed=seed,
             print_samples=False,
             use_stop_conditions=use_chi_sq_test,
             max_p_value_chi_sq=max_p_value_chi_sq)
        total_time_taken = timeit.default_timer() - start_time
        total_time_taken_list.append(total_time_taken)

        num_valid_attributes_list += num_valid_attributes_in_root
        num_nodes_pruned_list += num_nodes_prunned_per_fold
        tree_depth_list += max_depth_per_fold
        num_nodes_list += num_nodes_per_fold

        accuracy_with_missing_values_list.append(
            100.0 * num_correct_classifications_w_unkown / train_dataset.num_samples)
        try:
            accuracy_without_missing_values_list.append(
                100.0 * num_correct_classifications_wo_unkown
                / (train_dataset.num_samples - num_unkown))
        except ZeroDivisionError:
            pass
        num_unkown_list.append(num_unkown)

    save_fold_info(dataset_name, train_dataset.num_samples, num_trials, criterion.name,
                   min_num_samples_allowed, max_depth, use_chi_sq_test, max_p_value_chi_sq,
                   use_monte_carlo, criteria.ORDER_RANDOMLY, upper_p_value_threshold,
                   lower_p_value_threshold, prob_monte_carlo, np.array(num_valid_attributes_list),
                   np.array(num_nodes_list), np.array(tree_depth_list),
                   np.array(total_time_taken_list), np.array(accuracy_with_missing_values_list),
                   np.array(accuracy_without_missing_values_list), np.array(num_unkown_list),
                   np.array(num_nodes_pruned_list), output_split_char, output_file_descriptor)


def save_fold_info(dataset_name, num_total_samples, num_trials, criterion_name,
                   min_num_samples_allowed, max_depth, use_chi_sq_test, max_p_value_chi_sq,
                   use_monte_carlo, is_random_ordering, upper_p_value_threshold,
                   lower_p_value_threshold, prob_monte_carlo, num_valid_attributes_array,
                   num_nodes_array, tree_depth_array, total_time_taken_array,
                   accuracy_with_missing_values_array, accuracy_without_missing_values_array,
                   num_unkown_array, num_nodes_pruned_array, output_split_char,
                   output_file_descriptor):
    """Saves the experiment information in the CSV file.
    """
    assert num_trials > 0
    line_list = [str(datetime.datetime.now()),
                 dataset_name,
                 str(num_total_samples),
                 str(num_trials),
                 criterion_name,
                 str(min_num_samples_allowed),
                 str(max_depth),
                 str(NUM_FOLDS),

                 str(use_chi_sq_test),
                 str(max_p_value_chi_sq),

                 str(use_monte_carlo),
                 str(is_random_ordering),
                 str(upper_p_value_threshold),
                 str(lower_p_value_threshold),
                 str(prob_monte_carlo),

                 str(np.mean(num_valid_attributes_array)),
                 str(np.amax(num_valid_attributes_array)),
                 str(np.amin(num_valid_attributes_array)),

                 str(np.mean(num_nodes_array)),
                 str(np.amax(num_nodes_array)),
                 str(np.amin(num_nodes_array)),

                 str(np.mean(tree_depth_array)),
                 str(np.amax(tree_depth_array)),
                 str(np.amin(tree_depth_array)),

                 str(np.mean(total_time_taken_array)),

                 str(np.mean(accuracy_with_missing_values_array)),
                 str(np.mean(accuracy_without_missing_values_array)),
                 str(np.mean(num_unkown_array)),

                 str(np.mean(num_nodes_pruned_array))]

    print(output_split_char.join(line_list), file=output_file_descriptor)


def main(dataset_names, datasets_filepaths, key_attrib_indices, class_attrib_indices, split_chars,
         missing_value_strings, min_num_samples_allowed, max_depth, num_trials, use_chi_sq_test,
         max_p_value_chi_sq, use_monte_carlo, use_random_ordering, upper_p_value_threshold,
         lower_p_value_threshold, prob_monte_carlo, output_csv_filepath,
         output_split_char=OUTPUT_SPLIT_CHAR):
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
                                   min_num_samples_allowed,
                                   max_depth,
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
                                   min_num_samples_allowed,
                                   max_depth,
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
                                   min_num_samples_allowed,
                                   max_depth,
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

    # Poker Hand:
    DATASET_NAMES.append('Poker Hand')
    DATASETS_FILEPATHS.append(os.path.join(DATASET_BASE_PATH,
                                           'poker hand',
                                           'poker-hand-modified.csv'))
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
        'multiple_levels_experiment_1.csv')

    with open(OUTPUT_CSV_FILEPATH, 'a') as FOUT:
        FIELDS_LIST = ['Date Time',
                       'Dataset',
                       'Total Number of Samples',
                       'Number of Trials',
                       'Criterion',
                       'Number of Samples Forcing a Leaf',
                       'Maximum Depth Allowed',
                       'Number of folds',

                       'Uses Chi-Square Test',
                       'Maximum p-value Allowed by Chi-Square Test [between 0 and 1]',

                       'Uses Monte Carlo',
                       'Are Attributes in Random Order?',
                       'U [between 0 and 1]',
                       'L [between 0 and 1]',
                       'prob_monte_carlo [between 0 and 1]',

                       'Average Number of Valid Attributes in Root Node (m)',
                       'Maximum Number of Valid Attributes in Root Node (m)',
                       'Minimum Number of Valid Attributes in Root Node (m)',

                       'Average Number of Nodes (after prunning)',
                       'Maximum Number of Nodes (after prunning)',
                       'Minimum Number of Nodes (after prunning)',

                       'Average Tree Depth (after prunning)',
                       'Maximum Tree Depth (after prunning)',
                       'Minimum Tree Depth (after prunning)',

                       'Average Total Time Taken [s]',

                       'Average Accuracy Percentage (with missing values)',
                       'Average Accuracy Percentage (without missing values)',
                       'Average Number of Samples Classified using Unkown Value',

                       'Average Number of Nodes Pruned']

        print(OUTPUT_SPLIT_CHAR.join(FIELDS_LIST), file=FOUT)
        FOUT.flush()

    # (upper_p_value_threshold, lower_p_value_threshold, prob_monte_carlo)
    PARAMETERS_LIST = [(0.4, 0.1, 0.95),
                       (0.4, 0.1, 0.99),
                       (0.3, 0.1, 0.95),
                       (0.3, 0.1, 0.99)]

    USE_RANDOM = [False, True]
    MAX_P_VALUE_CHI_SQ = [0.1]
    MIN_NUM_SAMPLES_ALLOWED = [1]
    MAX_DEPTH = [5]

    for (curr_min_num_samples_allowed,
         curr_max_depth) in itertools.product(MIN_NUM_SAMPLES_ALLOWED,
                                              MAX_DEPTH):
        # Run without any bias treatment
        main(DATASET_NAMES,
             DATASETS_FILEPATHS,
             KEY_ATTRIB_INDICES,
             CLASS_ATTRIB_INDICES,
             SPLIT_CHARS,
             MISSING_VALUE_STRINGS,
             curr_min_num_samples_allowed,
             curr_max_depth,
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
                 curr_min_num_samples_allowed,
                 curr_max_depth,
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
                 curr_min_num_samples_allowed,
                 curr_max_depth,
                 NUM_TRIALS,
                 use_chi_sq_test=False,
                 max_p_value_chi_sq=None,
                 use_monte_carlo=True,
                 use_random_ordering=curr_use_random_ordering,
                 upper_p_value_threshold=curr_upper_p_value_threshold,
                 lower_p_value_threshold=curr_lower_p_value_threshold,
                 prob_monte_carlo=curr_prob_monte_carlo,
                 output_csv_filepath=OUTPUT_CSV_FILEPATH)
