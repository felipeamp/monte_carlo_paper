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
import monte_carlo

import numpy as np


#: Initial seed used in `random` and `numpy.random` modules.
RANDOM_SEED = 65537


def monte_carlo_experiment(dataset_name, train_dataset, criterion, min_num_samples_allowed,
                           max_depth, num_trials, use_chi_sq_test, max_p_value_chi_sq,
                           use_monte_carlo, upper_p_value_threshold, lower_p_value_threshold,
                           prob_monte_carlo, output_file_descriptor, output_split_char,
                           seed=RANDOM_SEED):
    """Runs `num_trials` experiments, each one doing a stratified cross-validation in `NUM_FOLDS`
    folds. Saves the training and classification information in the `output_file_descriptor` file.
    """
    for trial_number in range(num_trials):
        print('*'*80)
        print('STARTING TRIAL #{}'.format(trial_number + 1))
        print()

        # Resets the Monte Carlo caches for each tree trained.
        monte_carlo.clean_caches()

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
        accuracy_with_missing_values = (100.0 * num_correct_classifications_w_unkown
                                        / train_dataset.num_samples)
        try:
            accuracy_without_missing_values = (100.0 * num_correct_classifications_wo_unkown
                                               / (train_dataset.num_samples - num_unkown))
        except ZeroDivisionError:
            accuracy_without_missing_values = None

        save_trial_info(dataset_name, train_dataset.num_samples, trial_number + 1, criterion.name,
                        min_num_samples_allowed, max_depth, use_chi_sq_test, max_p_value_chi_sq,
                        use_monte_carlo, criteria.ORDER_RANDOMLY, upper_p_value_threshold,
                        lower_p_value_threshold, prob_monte_carlo,
                        np.mean(num_valid_attributes_in_root), np.mean(num_nodes_per_fold),
                        np.mean(max_depth_per_fold), total_time_taken, accuracy_with_missing_values,
                        accuracy_without_missing_values, num_unkown,
                        np.mean(num_nodes_prunned_per_fold), output_split_char,
                        output_file_descriptor)


def save_trial_info(dataset_name, num_total_samples, trial_number, criterion_name,
                    min_num_samples_allowed, max_depth, use_chi_sq_test, max_p_value_chi_sq,
                    use_monte_carlo, is_random_ordering, upper_p_value_threshold,
                    lower_p_value_threshold, prob_monte_carlo, avg_num_valid_attributes_in_root,
                    avg_num_nodes, avg_tree_depth, total_time_taken, accuracy_with_missing_values,
                    accuracy_without_missing_values, num_unkown, avg_num_nodes_pruned,
                    output_split_char, output_file_descriptor):
    """Saves the experiment information in the CSV file.
    """
    line_list = [str(datetime.datetime.now()),
                 dataset_name,
                 str(num_total_samples),
                 str(trial_number),
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

                 str(avg_num_valid_attributes_in_root),

                 str(avg_num_nodes),
                 str(avg_tree_depth),

                 str(total_time_taken),

                 str(accuracy_with_missing_values),
                 str(accuracy_without_missing_values),
                 str(num_unkown),

                 str(avg_num_nodes_pruned)]

    print(output_split_char.join(line_list), file=output_file_descriptor)
    output_file_descriptor.flush()


def main(dataset_names, datasets_filepaths, key_attrib_indices, class_attrib_indices, split_chars,
         missing_value_strings, min_num_samples_allowed, max_depth, num_trials, use_chi_sq_test,
         max_p_value_chi_sq, use_monte_carlo, use_random_ordering, upper_p_value_threshold,
         lower_p_value_threshold, prob_monte_carlo, output_csv_filepath,
         output_split_char=','):
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
            print('-'*100)
            print('Twoing')
            print()
            monte_carlo_experiment(dataset_names[dataset_number],
                                   train_dataset,
                                   criteria.Twoing(),
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
            print('-'*100)
            print('Gain Ratio')
            print()
            monte_carlo_experiment(dataset_names[dataset_number],
                                   train_dataset,
                                   criteria.GainRatio(),
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

def init_datasets_info():
    """Load information about every dataset available.

    Returns:
        Tuple containing, in order:
            - Name of each dataset;
            - Filepath of each dataset;
            - Key's attribute index in each dataset (may be `None`);
            - Class' attribute index in each dataset (may be negative, indexing from the end);
            - Character used to split csv entries in each dataset;
            - String representing missing values in each dataset (may be `None`).
    """
    dataset_names = []
    datasets_filepaths = []
    key_attrib_indices = []
    class_attrib_indices = []
    split_chars = []
    missing_value_strings = []

    dataset_base_path = os.path.join('.', 'datasets')

    # Adult census income
    dataset_names.append('Adult Census Income')
    datasets_filepaths.append(os.path.join(dataset_base_path,
                                           'adult census income',
                                           'adult_no_quotation_marks.csv'))
    key_attrib_indices.append(None)
    class_attrib_indices.append(-1)
    split_chars.append(',')
    missing_value_strings.append('')

    # Cars:
    dataset_names.append('Cars')
    datasets_filepaths.append(os.path.join(dataset_base_path,
                                           'cars'
                                           'cars.csv'))
    key_attrib_indices.append(None)
    class_attrib_indices.append(-1)
    split_chars.append(',')
    missing_value_strings.append(None)

    # Contraceptive:
    dataset_names.append('Contraceptive')
    datasets_filepaths.append(os.path.join(dataset_base_path,
                                           'contraceptive'
                                           'contraceptive.csv'))
    key_attrib_indices.append(None)
    class_attrib_indices.append(-1)
    split_chars.append(',')
    missing_value_strings.append(None)

    # Mushroom
    dataset_names.append('Mushroom')
    datasets_filepaths.append(os.path.join(dataset_base_path,
                                           'mushroom',
                                           'agaricus-lepiota.csv'))
    key_attrib_indices.append(None)
    class_attrib_indices.append(0)
    split_chars.append(',')
    missing_value_strings.append('?')

    # Nursery:
    dataset_names.append('Nursery')
    datasets_filepaths.append(os.path.join(dataset_base_path,
                                           'nursery',
                                           'nursery.txt'))
    key_attrib_indices.append(None)
    class_attrib_indices.append(-1)
    split_chars.append(',')
    missing_value_strings.append(None)

    # Poker Hand:
    dataset_names.append('Poker Hand')
    datasets_filepaths.append(os.path.join(dataset_base_path,
                                           'poker hand',
                                           'poker-hand-modified.csv'))
    key_attrib_indices.append(None)
    class_attrib_indices.append(-1)
    split_chars.append(',')
    missing_value_strings.append(None)

    # Splice Junction:
    dataset_names.append('Splice Junction')
    datasets_filepaths.append(os.path.join(dataset_base_path,
                                           'splice junction'
                                           'splice-junction-modified.csv'))
    key_attrib_indices.append(1)
    class_attrib_indices.append(0)
    split_chars.append(',')
    missing_value_strings.append(None)

    # Titanic Survive:
    dataset_names.append('Titanic Survive')
    datasets_filepaths.append(os.path.join(dataset_base_path,
                                           'titanic',
                                           'titanicSurvive.csv'))
    key_attrib_indices.append(None)
    class_attrib_indices.append(-1)
    split_chars.append(',')
    missing_value_strings.append(None)

    return (dataset_names,
            datasets_filepaths,
            key_attrib_indices,
            class_attrib_indices,
            split_chars,
            missing_value_strings)

def init_output_csv(output_csv_filepath, output_split_char=','):
    """Writes the header to the CSV output file.
    """
    with open(output_csv_filepath, 'a') as fout:
        fields_list = ['Date Time',
                       'Dataset',
                       'Total Number of Samples',
                       'Trial Number',
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

                       'Average Number of Nodes (after prunning)',
                       'Average Tree Depth (after prunning)',

                       'Total Time Taken for Cross-Validation [s]',

                       'Accuracy Percentage (with missing values)',
                       'Accuracy Percentage (without missing values)',
                       'Number of Samples Classified using Unkown Value',

                       'Average Number of Nodes Pruned']
        print(output_split_char.join(fields_list), file=fout)
        fout.flush()


if __name__ == '__main__':
    # Datasets
    (DATASET_NAMES,
     DATASETS_FILEPATHS,
     KEY_ATTRIB_INDICES,
     CLASS_ATTRIB_INDICES,
     SPLIT_CHARS,
     MISSING_VALUE_STRINGS) = init_datasets_info()

    # Output
    OUTPUT_CSV_FILEPATH = os.path.join(
        '.',
        'outputs',
        'multiple_levels_experiment_2.csv')
    init_output_csv(OUTPUT_CSV_FILEPATH)

    # Parameters configurations

    # Minimum number of samples allowed in a TreeNode and still allowing it to split during training
    MIN_NUM_SAMPLES_ALLOWED = 1
    NUM_FOLDS = 10 # Number of folds to be used during cross-validation.
    NUM_TRIALS = 5 # Number of experiments to be done with each parameters combination.

    MAX_P_VALUE_CHI_SQ = [0.1]
    MAX_DEPTH = [5]

    # (upper_p_value_threshold, lower_p_value_threshold, prob_monte_carlo)
    PARAMETERS_LIST = [(0.4, 0.1, 0.95),
                       (0.4, 0.1, 0.99),
                       (0.3, 0.1, 0.95),
                       (0.3, 0.1, 0.99)]
    USE_RANDOM = [False, True]

    # Experiments
    for curr_max_depth in MAX_DEPTH:
        # Run without any bias treatment
        main(DATASET_NAMES,
             DATASETS_FILEPATHS,
             KEY_ATTRIB_INDICES,
             CLASS_ATTRIB_INDICES,
             SPLIT_CHARS,
             MISSING_VALUE_STRINGS,
             MIN_NUM_SAMPLES_ALLOWED,
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
        # Run with Chi Square test and minimum number of samples in second most frequent value
        for curr_max_p_value_chi_sq in MAX_P_VALUE_CHI_SQ:
            main(DATASET_NAMES,
                 DATASETS_FILEPATHS,
                 KEY_ATTRIB_INDICES,
                 CLASS_ATTRIB_INDICES,
                 SPLIT_CHARS,
                 MISSING_VALUE_STRINGS,
                 MIN_NUM_SAMPLES_ALLOWED,
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
                 MIN_NUM_SAMPLES_ALLOWED,
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
