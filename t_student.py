#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to calculate t-student statistics of experiments.
'''

import collections
import itertools
import json
import math
import os
import statistics
import sys

from scipy.stats import t as student_t


ColumnIndices = collections.namedtuple('ColumnIndices',
                                       ['dataset_col',
                                        'criterion_col',
                                        'use_chi_sq',
                                        'max_chi_sq_p_value',
                                        'min_num_values_second_most_freq_value',
                                        'num_min_samples_allowed',
                                        'use_second_largest_class_min_samples',
                                        'min_samples_in_second_largest_class',
                                        'use_monte_carlo',
                                        'upper_p_value_threshold',
                                        'lower_p_value_threshold',
                                        'prob_monte_carlo',
                                        'use_random_order',
                                        'use_one_attrib_per_num_values',
                                        'trial_number_col',
                                        'accuracy_w_missing_col',
                                        'accuracy_wo_missing_col',
                                        'num_nodes_col'])

#: Contain the column indices for a cross-validation experiment. Starts at zero.
CROSS_VALIDATION_COLUMN_INDICES = ColumnIndices(dataset_col=1,
                                                criterion_col=4,
                                                use_chi_sq=11,
                                                max_chi_sq_p_value=12,
                                                min_num_values_second_most_freq_value=13,
                                                num_min_samples_allowed=8,
                                                use_second_largest_class_min_samples=9,
                                                min_samples_in_second_largest_class=10,
                                                use_monte_carlo=14,
                                                upper_p_value_threshold=16,
                                                lower_p_value_threshold=17,
                                                prob_monte_carlo=18,
                                                use_random_order=15,
                                                use_one_attrib_per_num_values=19,
                                                trial_number_col=3,
                                                accuracy_w_missing_col=24,
                                                accuracy_wo_missing_col=25,
                                                num_nodes_col=32)

#: Contain the column indices for a train-and-test experiment. Starts at zero.
TRAIN_AND_TEST_COLUMN_INDICES = ColumnIndices(dataset_col=1,
                                              criterion_col=6,
                                              use_chi_sq=11,
                                              max_chi_sq_p_value=12,
                                              min_num_values_second_most_freq_value=13,
                                              num_min_samples_allowed=8,
                                              use_second_largest_class_min_samples=9,
                                              min_samples_in_second_largest_class=10,
                                              use_monte_carlo=14,
                                              upper_p_value_threshold=16,
                                              lower_p_value_threshold=17,
                                              prob_monte_carlo=18,
                                              use_random_order=15,
                                              use_one_attrib_per_num_values=19,
                                              trial_number_col=5,
                                              accuracy_w_missing_col=35,
                                              accuracy_wo_missing_col=36,
                                              num_nodes_col=39)


PruningParameters = collections.namedtuple('PruningParameters',
                                           ['use_chi_sq',
                                            'max_chi_sq_p_value',
                                            'min_num_values_second_most_freq_value',
                                            'num_min_samples_allowed',
                                            'use_second_largest_class_min_samples',
                                            'min_samples_in_second_largest_class',
                                            'use_monte_carlo',
                                            'upper_p_value_threshold',
                                            'lower_p_value_threshold',
                                            'prob_monte_carlo',
                                            'use_random_order',
                                            'use_one_attrib_per_num_values'])

def main(output_path):
    '''Calculates the t-student statistics of experiments contained in this folder.

    The `output_path` folder must contain the `raw_output.csv` file and the `experiment_config.json`
    file, otherwise the function will exit.
    '''
    raw_output_path = os.path.join(output_path, 'raw_output.csv')
    if (not os.path.exists(raw_output_path)
            or not os.path.isfile(raw_output_path)):
        print('This path does not contain the output of an experiment.')
        sys.exit(1)

    experiment_config_filepath = os.path.join(output_path, 'experiment_config.json')
    if (not os.path.exists(experiment_config_filepath)
            or not os.path.isfile(experiment_config_filepath)):
        print('This path does not contain the output of an experiment.')
        sys.exit(1)
    with open(experiment_config_filepath, 'r') as experiment_config_json:
        experiment_config = json.load(experiment_config_json)

    if experiment_config["use cross-validation"]:
        column_indices = CROSS_VALIDATION_COLUMN_INDICES
    else:
        column_indices = TRAIN_AND_TEST_COLUMN_INDICES

    single_sided_p_value_threshold = experiment_config["t-test single-sided p-value"]

    raw_data = _load_raw_data(raw_output_path, column_indices)
    _save_raw_stats(raw_data, output_path)
    _save_aggreg_stats(output_path, single_sided_p_value_threshold)


def _load_raw_data(raw_output_path, column_indices):
    def _init_raw_data():
        # This function creates (in a lazy way) an infinitely-nested default dict. This is
        # useful when creating a default dict highly nested.
        return collections.defaultdict(_init_raw_data)

    raw_data = _init_raw_data()
    has_read_header = False
    with open(raw_output_path, 'r') as fin:
        for line in fin:
            if not has_read_header:
                has_read_header = True
                continue
            line_list = line.rstrip().split(',')

            dataset_name = line_list[column_indices.dataset_col]
            criterion_name = line_list[column_indices.criterion_col]
            trial_number = line_list[column_indices.trial_number_col]

            pruning_parameters_list = []
            for column_index in column_indices[2:14]:
                pruning_parameters_list.append(line_list[column_index])
            pruning_parameters = PruningParameters(*pruning_parameters_list)

            accuracy_w_missing = float(line_list[column_indices.accuracy_w_missing_col])
            try:
                accuracy_wo_missing = float(line_list[column_indices.accuracy_wo_missing_col])
            except ValueError:
                accuracy_wo_missing = None
            num_nodes = float(line_list[column_indices.num_nodes_col])

            raw_data[dataset_name][criterion_name][pruning_parameters][trial_number] = (
                accuracy_w_missing,
                accuracy_wo_missing,
                num_nodes)
    return raw_data


def _save_raw_stats(raw_data, output_path):
    raw_stats_output_file = os.path.join(output_path, 'raw_t_student_stats.csv')
    with open(raw_stats_output_file, 'w') as fout:
        header = ['Dataset',
                  'Criterion Name',

                  'use_chi_sq (1)',
                  'max_chi_sq_p_value (1)',
                  'min_num_values_second_most_freq_value (1)',
                  'num_min_samples_allowed (1)',
                  'use_second_largest_class_min_samples (1)',
                  'min_samples_in_second_largest_class (1)',
                  'use_monte_carlo (1)',
                  'upper_p_value_threshold (1)',
                  'lower_p_value_threshold (1)',
                  'prob_monte_carlo (1)',
                  'use_random_order (1)',
                  'use_one_attrib_per_num_values (1)',

                  'use_chi_sq (2)',
                  'max_chi_sq_p_value (2)',
                  'min_num_values_second_most_freq_value (2)',
                  'num_min_samples_allowed (2)',
                  'use_second_largest_class_min_samples (2)',
                  'min_samples_in_second_largest_class (2)',
                  'use_monte_carlo (2)',
                  'upper_p_value_threshold (2)',
                  'lower_p_value_threshold (2)',
                  'prob_monte_carlo (2)',
                  'use_random_order (2)',
                  'use_one_attrib_per_num_values (2)',

                  'Paired t-statistics on Accuracy with Missing Values',
                  'Degrees of Freedom of Accuracy with Missing Values',
                  'P-value t-statistics on Accuracy with Missing Values',
                  'Paired t-statistics on Accuracy without Missing Values',
                  'Degrees of Freedom of Accuracy without Missing Values',
                  'P-value t-statistics on Accuracy without Missing Values',
                  'Paired t-statistics on Number of Nodes',
                  'Degrees of Freedom of Number of Nodes',
                  'P-value t-statistics on Number of Nodes']
        print(','.join(header), file=fout)

        for dataset_name in raw_data:
            for criterion_name in raw_data[dataset_name]:
                for (prunning_parameters_1,
                     prunning_parameters_2) in itertools.combinations(
                         raw_data[dataset_name][criterion_name], 2):

                    accuracy_w_missing_diff = []
                    accuracy_wo_missing_diff = []
                    num_nodes_diff = []

                    trial_number_intersection = (
                        set(raw_data[dataset_name][criterion_name][prunning_parameters_1].keys())
                        & set(raw_data[dataset_name][criterion_name][prunning_parameters_2].keys()))
                    for trial_number in trial_number_intersection:
                        prunning_parameters_1_data = raw_data[dataset_name][criterion_name][
                            prunning_parameters_1][trial_number]
                        prunning_parameters_2_data = raw_data[dataset_name][criterion_name][
                            prunning_parameters_2][trial_number]

                        accuracy_w_missing_diff.append(
                            prunning_parameters_1_data[0] - prunning_parameters_2_data[0])
                        if (prunning_parameters_1_data[1] is not None
                                and prunning_parameters_2_data[1] is not None):
                            accuracy_wo_missing_diff.append(
                                prunning_parameters_1_data[1] - prunning_parameters_2_data[1])
                        num_nodes_diff.append(
                            prunning_parameters_1_data[2] - prunning_parameters_2_data[2])

                    (t_statistic_w_missing,
                     p_value_w_missing) = _calculate_t_statistic(accuracy_w_missing_diff)
                    (t_statistic_wo_missing,
                     p_value_wo_missing) = _calculate_t_statistic(accuracy_wo_missing_diff)
                    (t_statistic_num_nodes,
                     p_value_num_nodes) = _calculate_t_statistic(num_nodes_diff)
                    print(','.join([dataset_name,
                                    criterion_name,
                                    *list(map(str, prunning_parameters_1)),
                                    *list(map(str, prunning_parameters_2)),
                                    str(t_statistic_w_missing),
                                    str(len(accuracy_w_missing_diff) - 1),
                                    str(p_value_w_missing),
                                    str(t_statistic_wo_missing),
                                    str(len(accuracy_wo_missing_diff) - 1),
                                    str(p_value_wo_missing),
                                    str(t_statistic_num_nodes),
                                    str(len(num_nodes_diff) - 1),
                                    str(p_value_num_nodes)]),
                          file=fout)


def _calculate_t_statistic(samples_list):
    if len(samples_list) <= 1:
        return None, None
    mean = statistics.mean(samples_list)
    variance = statistics.variance(samples_list)
    if variance == 0.0:
        # Every sample has the same value.
        if mean == 0.0:
            return 0.0, 0.5
        if mean > 0.0:
            return float('+inf'), 0.0
        return float('-inf'), 1.0

    num_samples = len(samples_list)
    t_statistic = mean / math.sqrt(variance / num_samples)
    degrees_of_freedom = num_samples - 1
    p_value = 1. - student_t.cdf(t_statistic, degrees_of_freedom)
    return t_statistic, p_value


def _save_aggreg_stats(output_path, single_sided_p_value_threshold):
    # aggreg_data[(dataset,
    #              criterion_name,
    #              prunning_parameters)] = [num_times_stat_better_w_missing,
    #                                       num_times_stat_better_wo_missing,
    #                                       num_times_stat_larger_num_nodes]
    aggreg_data = {}
    raw_stats_output_file = os.path.join(output_path, 'raw_t_student_stats.csv')
    has_read_header = False
    with open(raw_stats_output_file, 'r') as fin:
        for line in fin:
            if not has_read_header:
                has_read_header = True
                continue
            line_list = line.rstrip().split(',')

            dataset_name = line_list[0]
            criterion_name = line_list[1]

            prunning_parameters_1 = PruningParameters(*line_list[2:14])
            prunning_parameters_2 = PruningParameters(*line_list[14:26])

            if (dataset_name, criterion_name, prunning_parameters_1) not in aggreg_data:
                aggreg_data[(dataset_name,
                             criterion_name,
                             prunning_parameters_1)] = [0, 0, 0, 0, 0, 0]
            if (dataset_name, criterion_name, prunning_parameters_2) not in aggreg_data:
                aggreg_data[(dataset_name,
                             criterion_name,
                             prunning_parameters_2)] = [0, 0, 0, 0, 0, 0]

            try:
                p_value_w_missing = float(line_list[28])
                if p_value_w_missing <= single_sided_p_value_threshold:
                    aggreg_data[(dataset_name,
                                 criterion_name,
                                 prunning_parameters_1)][0] += 1
                    aggreg_data[(dataset_name,
                                 criterion_name,
                                 prunning_parameters_2)][3] += 1
                elif p_value_w_missing >= 1. - single_sided_p_value_threshold:
                    aggreg_data[(dataset_name,
                                 criterion_name,
                                 prunning_parameters_1)][3] += 1
                    aggreg_data[(dataset_name,
                                 criterion_name,
                                 prunning_parameters_2)][0] += 1
            except ValueError:
                pass

            try:
                p_value_wo_missing = float(line_list[31])
                if p_value_wo_missing is not None:
                    if p_value_wo_missing <= single_sided_p_value_threshold:
                        aggreg_data[(dataset_name,
                                     criterion_name,
                                     prunning_parameters_1)][1] += 1
                        aggreg_data[(dataset_name,
                                     criterion_name,
                                     prunning_parameters_2)][4] += 1
                    elif p_value_wo_missing >= 1. - single_sided_p_value_threshold:
                        aggreg_data[(dataset_name,
                                     criterion_name,
                                     prunning_parameters_1)][4] += 1
                        aggreg_data[(dataset_name,
                                     criterion_name,
                                     prunning_parameters_2)][1] += 1
            except ValueError:
                pass

            try:
                p_value_num_nodes = float(line_list[34])
                if p_value_num_nodes is not None:
                    if p_value_num_nodes <= single_sided_p_value_threshold:
                        aggreg_data[(dataset_name,
                                     criterion_name,
                                     prunning_parameters_1)][2] += 1
                        aggreg_data[(dataset_name,
                                     criterion_name,
                                     prunning_parameters_2)][5] += 1
                    elif p_value_num_nodes >= 1. - single_sided_p_value_threshold:
                        aggreg_data[(dataset_name,
                                     criterion_name,
                                     prunning_parameters_1)][5] += 1
                        aggreg_data[(dataset_name,
                                     criterion_name,
                                     prunning_parameters_2)][2] += 1
            except ValueError:
                pass

    aggreg_stats_output_file = os.path.join(output_path, 'aggreg_t_student_stats.csv')
    with open(aggreg_stats_output_file, 'w') as fout:
        header = ['Dataset',
                  'Criterion Name',

                  'use_chi_sq',
                  'max_chi_sq_p_value',
                  'min_num_values_second_most_freq_value',
                  'num_min_samples_allowed',
                  'use_second_largest_class_min_samples',
                  'min_samples_in_second_largest_class',
                  'use_monte_carlo',
                  'upper_p_value_threshold',
                  'lower_p_value_threshold',
                  'prob_monte_carlo',
                  'use_random_order',
                  'use_one_attrib_per_num_values',

                  'Number of times is statistically better with missing values',
                  'Number of times is statistically better without missing values',
                  'Number of times has statistically larger number of nodes',
                  'Number of times is statistically worse with missing values',
                  'Number of times is statistically worse without missing values',
                  'Number of times has statistically smaller number of nodes']
        print(','.join(header), file=fout)
        for keys in sorted(aggreg_data):
            values = map(str, aggreg_data[keys])
            print(','.join([keys[0], keys[1], *keys[2], *values]), file=fout)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please include a path to an experiment output folder.')
        sys.exit(1)

    main(sys.argv[1].replace(r'\ ', ' '))
