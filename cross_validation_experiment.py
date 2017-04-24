#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to run cross-validation tests with decision trees, usually with maximum depth >= 2.
'''


import datetime
import itertools
import os
import sys
import timeit

import criteria
import dataset
import decision_tree
import monte_carlo

import numpy as np


#: Initial seed used in `random` and `numpy.random` modules.
RANDOM_SEED = 65537


def main(experiment_config):
    """Sets the configurations according to `experiment_config` and runs them.
    """
    raw_output_filepath = os.path.join(experiment_config["output folder"], 'raw_output.csv')
    with open(raw_output_filepath, 'a') as fout:
        init_raw_output_csv(fout, output_split_char=',')
        criteria_list = get_criteria(experiment_config["criteria"])

        if experiment_config["prunning parameters"]["use chi-sq test"]:
            max_p_value_chi_sq = experiment_config["prunning parameters"]["max chi-sq p-value"]
            decision_tree.MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE = experiment_config[
                "second most freq value min samples"]
        else:
            max_p_value_chi_sq = None

        if experiment_config["prunning parameters"]["use monte carlo"]:
            upper_p_value_threshold = experiment_config[
                "prunning parameters"]["monte carlo parameters"]["upper p-value threshold"]
            lower_p_value_threshold = experiment_config[
                "prunning parameters"]["monte carlo parameters"]["lower p-value threshold"]
            prob_monte_carlo = experiment_config[
                "prunning parameters"]["monte carlo parameters"]["prob monte carlo"]
            random_oder_list = experiment_config[
                "prunning parameters"]["monte carlo parameters"]["use random order"]
        else:
            upper_p_value_threshold = None
            lower_p_value_threshold = None
            prob_monte_carlo = None
            random_oder_list = None

        decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS = experiment_config[
            "use second most freq class min samples"]
        if decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS:
            decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS = experiment_config[
                "second most freq class min samples"]
        else:
            decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS = None

        if experiment_config["use all datasets"]:
            datasets_configs = dataset.load_all_configs(experiment_config["datasets basepath"])
            datasets = dataset.load_all_datasets(datasets_configs)

            for ((dataset_name, curr_dataset),
                 use_random_ordering,
                 min_num_samples_allowed) in itertools.product(
                     datasets,
                     random_oder_list,
                     experiment_config["num min samples allowed"]):
                criteria.ORDER_RANDOMLY = use_random_ordering
                for criterion in criteria_list:
                    print('-'*100)
                    print(criterion.name)
                    print()
                    run(dataset_name,
                        curr_dataset,
                        criterion,
                        min_num_samples_allowed=min_num_samples_allowed,
                        max_depth=experiment_config["max depth"],
                        num_trials=experiment_config["num trials"],
                        num_folds=experiment_config["num folds"],
                        is_stratified=experiment_config["is stratified"],
                        use_chi_sq_test=experiment_config["prunning parameters"]["use chi-sq test"],
                        max_p_value_chi_sq=max_p_value_chi_sq,
                        use_monte_carlo=experiment_config["prunning parameters"]["use monte carlo"],
                        upper_p_value_threshold=upper_p_value_threshold,
                        lower_p_value_threshold=lower_p_value_threshold,
                        prob_monte_carlo=prob_monte_carlo,
                        output_file_descriptor=fout,
                        output_split_char=',')
        else:
            datasets_folders = [os.path.join(experiment_config["datasets basepath"], folderpath)
                                for folderpath in experiment_config["datasets folders"]]
            datasets_configs = [dataset.load_config(folderpath)
                                for folderpath in datasets_folders]
            datasets_configs.sort(key=lambda x: x["dataset name"].lower())

            for (dataset_config,
                 use_random_ordering,
                 min_num_samples_allowed) in itertools.product(
                     datasets,
                     random_oder_list,
                     experiment_config["min num samples allowed"]):
                curr_dataset = dataset.Dataset(dataset_config["filepath"],
                                               dataset_config["key attrib index"],
                                               dataset_config["class attrib index"],
                                               dataset_config["split char"],
                                               dataset_config["missing value string"])
                criteria.ORDER_RANDOMLY = use_random_ordering
                for criterion in criteria_list:
                    print('-'*100)
                    print(criterion.name)
                    print()
                    run(dataset_config["dataset name"],
                        curr_dataset,
                        criterion,
                        min_num_samples_allowed=min_num_samples_allowed,
                        max_depth=experiment_config["max depth"],
                        num_trials=experiment_config["num trials"],
                        num_folds=experiment_config["num folds"],
                        is_stratified=experiment_config["is stratified"],
                        use_chi_sq_test=experiment_config["prunning parameters"]["use chi-sq test"],
                        max_p_value_chi_sq=max_p_value_chi_sq,
                        use_monte_carlo=experiment_config["prunning parameters"]["use monte carlo"],
                        upper_p_value_threshold=upper_p_value_threshold,
                        lower_p_value_threshold=lower_p_value_threshold,
                        prob_monte_carlo=prob_monte_carlo,
                        output_file_descriptor=fout,
                        output_split_char=',')


def init_raw_output_csv(raw_output_file_descriptor, output_split_char=','):
    """Writes the header to the raw output CSV file.
    """
    fields_list = ['Date Time',
                   'Dataset',
                   'Total Number of Samples',
                   'Trial Number',
                   'Criterion',
                   'Maximum Depth Allowed',
                   'Number of folds',
                   'Is stratified?',

                   'Number of Samples Forcing a Leaf',
                   'Use Min Samples in Second Largest Class?',
                   'Min Samples in Second Largest Class',

                   'Uses Chi-Square Test',
                   'Maximum p-value Allowed by Chi-Square Test [between 0 and 1]',
                   'Minimum Number in Second Most Frequent Value',

                   'Uses Monte Carlo',
                   'Are Attributes in Random Order?',
                   'U [between 0 and 1]',
                   'L [between 0 and 1]',
                   'prob_monte_carlo [between 0 and 1]',

                   'Average Number of Valid Attributes in Root Node (m)',

                   'Average Number of Nodes (after prunning)',
                   'Average Tree Depth (after prunning)',

                   'Total Time Taken for Cross-Validation [s]',

                   'Accuracy Percentage on Trivial Trees (with no splits)',

                   'Accuracy Percentage (with missing values)',
                   'Accuracy Percentage (without missing values)',
                   'Number of Samples Classified using Unkown Value',

                   'Average Number of Nodes Pruned']
    print(output_split_char.join(fields_list), file=raw_output_file_descriptor)
    raw_output_file_descriptor.flush()


def get_criteria(criteria_names_list):
    """Given a list of criteria names, returns a list of all there criteria (as `Criterion`'s).
    If a criterion name is unkown, the system will exit the experiment.
    """
    criteria_list = []
    for criterion_name in criteria_names_list:
        if criterion_name == "Gini Gain":
            criteria_list.append(criteria.GiniGain())
        elif criterion_name == "Twoing":
            criteria_list.append(criteria.Twoing())
        elif criterion_name == "Gain Ratio":
            criteria_list.append(criteria.GainRatio())
        else:
            print('Unkown criterion name:', criterion_name)
            print('Exiting.')
            sys.exit(1)
    return criteria_list


def run(dataset_name, train_dataset, criterion, min_num_samples_allowed, max_depth, num_trials,
        num_folds, is_stratified, use_chi_sq_test, max_p_value_chi_sq, use_monte_carlo,
        upper_p_value_threshold, lower_p_value_threshold, prob_monte_carlo, output_file_descriptor,
        output_split_char=',', seed=RANDOM_SEED):
    """Runs `num_trials` experiments, each one doing a stratified cross-validation in `num_folds`
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
         num_valid_attributes_in_root,
         trivial_accuracy_percentage) = tree.cross_validate(
             dataset=train_dataset,
             num_folds=num_folds,
             max_depth=max_depth,
             min_samples_per_node=min_num_samples_allowed,
             is_stratified=is_stratified,
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
                        max_depth, num_folds, is_stratified, min_num_samples_allowed,
                        decision_tree.USE_MIN_SAMPLES_SECOND_LARGEST_CLASS,
                        decision_tree.MIN_SAMPLES_SECOND_LARGEST_CLASS,
                        use_chi_sq_test, max_p_value_chi_sq,
                        decision_tree.MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE, use_monte_carlo,
                        criteria.ORDER_RANDOMLY, upper_p_value_threshold, lower_p_value_threshold,
                        prob_monte_carlo, np.mean(num_valid_attributes_in_root),
                        np.mean(num_nodes_per_fold), np.mean(max_depth_per_fold), total_time_taken,
                        trivial_accuracy_percentage, accuracy_with_missing_values,
                        accuracy_without_missing_values, num_unkown,
                        np.mean(num_nodes_prunned_per_fold), output_split_char,
                        output_file_descriptor)


def save_trial_info(dataset_name, num_total_samples, trial_number, criterion_name,
                    max_depth, num_folds, is_stratified, min_num_samples_allowed,
                    use_min_samples_second_largest_class, min_samples_second_largest_class,
                    use_chi_sq_test, max_p_value_chi_sq, min_num_second_most_freq_value,
                    use_monte_carlo, is_random_ordering, upper_p_value_threshold,
                    lower_p_value_threshold, prob_monte_carlo, avg_num_valid_attributes_in_root,
                    avg_num_nodes, avg_tree_depth, total_time_taken, trivial_accuracy_percentage,
                    accuracy_with_missing_values, accuracy_without_missing_values, num_unkown,
                    avg_num_nodes_pruned, output_split_char, output_file_descriptor):
    """Saves the experiment's trial information in the CSV file.
    """
    line_list = [str(datetime.datetime.now()),
                 dataset_name,
                 str(num_total_samples),
                 str(trial_number),
                 criterion_name,
                 str(max_depth),
                 str(num_folds),
                 str(is_stratified),

                 str(min_num_samples_allowed),
                 str(use_min_samples_second_largest_class),
                 str(min_samples_second_largest_class),

                 str(use_chi_sq_test),
                 str(max_p_value_chi_sq),
                 str(min_num_second_most_freq_value),

                 str(use_monte_carlo),
                 str(is_random_ordering),
                 str(upper_p_value_threshold),
                 str(lower_p_value_threshold),
                 str(prob_monte_carlo),

                 str(avg_num_valid_attributes_in_root),

                 str(avg_num_nodes),
                 str(avg_tree_depth),

                 str(total_time_taken),

                 str(trivial_accuracy_percentage),
                 str(accuracy_with_missing_values),
                 str(accuracy_without_missing_values),
                 str(num_unkown),

                 str(avg_num_nodes_pruned)]

    print(output_split_char.join(line_list), file=output_file_descriptor)
    output_file_descriptor.flush()
