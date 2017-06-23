#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to calculate the difference in p-values between precise monte
carlo criteria and their counterparts using the Monte Carlo framework.
'''

import collections
import os
import sys


PValues = collections.namedtuple('PValues',
                                 ['dataset_col',
                                  'criterion_col',
                                  'trial_number_col',
                                  'attribute_name',
                                  'attribute_position',
                                  'criterion_value',
                                  'p_value'])


#: Contain the column indices for the p-values raw_output. Starts at zero.
P_VALUES_INDICES = PValues(dataset_col=1,
                           criterion_col=4,
                           trial_number_col=3,
                           attribute_name=7,
                           attribute_position=9,
                           criterion_value=10,
                           p_value=11)


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
                                        'position_accepted_attribute'])

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
                                              position_accepted_attribute=28)


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
    '''Calculates the p-values differences for the current experiment. `output_path`
    must contain a raw_output.csv file containing the Monte Carlo framework experiments
    and a raw_p_value_output.csv file containing all the precise p-values.
    '''
    raw_output_path = os.path.join(output_path, 'raw_output.csv')
    if (not os.path.exists(raw_output_path)
            or not os.path.isfile(raw_output_path)):
        print('This path does not contain the raw output of an experiment.')
        sys.exit(1)

    raw_p_value_output_path = os.path.join(output_path, 'raw_p_value_output.csv')
    if (not os.path.exists(raw_p_value_output_path)
            or not os.path.isfile(raw_p_value_output_path)):
        print('This path does not contain the raw p-value output of an experiment.')
        sys.exit(1)

    raw_data = _load_raw_data(raw_output_path)
    raw_p_value_data = _load_raw_p_value_data(raw_p_value_output_path)
    _save_p_value_stats(raw_data, raw_p_value_data, output_path)


def _load_raw_data(raw_output_path):
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
            if not line:
                continue
            line_list = line.rstrip().split(',')

            dataset_name = line_list[TRAIN_AND_TEST_COLUMN_INDICES.dataset_col]
            criterion_name = line_list[TRAIN_AND_TEST_COLUMN_INDICES.criterion_col]
            trial_number = int(line_list[TRAIN_AND_TEST_COLUMN_INDICES.trial_number_col])

            pruning_parameters_list = []
            for column_index in TRAIN_AND_TEST_COLUMN_INDICES[2:14]:
                pruning_parameters_list.append(line_list[column_index])
            pruning_parameters = PruningParameters(*pruning_parameters_list)
            if (pruning_parameters.use_chi_sq != 'False' or
                    pruning_parameters.num_min_samples_allowed != '1' or
                    pruning_parameters.use_second_largest_class_min_samples != 'False' or
                    pruning_parameters.use_monte_carlo != 'True' or
                    pruning_parameters.upper_p_value_threshold != '0.4' or
                    pruning_parameters.lower_p_value_threshold != '0.1' or
                    pruning_parameters.prob_monte_carlo != '0.95' or
                    pruning_parameters.use_random_order != 'False' or
                    pruning_parameters.use_one_attrib_per_num_values != 'False'):
                continue

            position_accepted_attribute = line_list[
                TRAIN_AND_TEST_COLUMN_INDICES.position_accepted_attribute]

            raw_data[dataset_name][criterion_name][
                trial_number] = position_accepted_attribute
    return raw_data


def _load_raw_p_value_data(raw_p_value_output_path):
    def _init_raw_p_value():
        # This function creates (in a lazy way) an infinitely-nested default dict. This is
        # useful when creating a default dict highly nested.
        return collections.defaultdict(_init_raw_p_value)

    raw_p_value_data = _init_raw_p_value()
    has_read_header = False
    with open(raw_p_value_output_path, 'r') as fin:
        for line in fin:
            if not has_read_header:
                has_read_header = True
                continue
            if not line:
                continue
            line_list = line.rstrip().split(',')

            dataset_name = line_list[P_VALUES_INDICES.dataset_col]
            criterion_name = line_list[P_VALUES_INDICES.criterion_col]
            trial_number = int(line_list[P_VALUES_INDICES.trial_number_col])
            attribute_name = line_list[P_VALUES_INDICES.attribute_name]
            attribute_position = line_list[P_VALUES_INDICES.attribute_position]
            p_value = float(line_list[P_VALUES_INDICES.p_value])

            raw_p_value_data[dataset_name][criterion_name][trial_number][
                attribute_position] = (attribute_name, p_value, attribute_position)
    return raw_p_value_data

def _save_p_value_stats(raw_data, raw_p_value_data, output_path):
    p_value_stats_output_file = os.path.join(output_path, 'p_value_stats.csv')
    with open(p_value_stats_output_file, 'w') as fout:
        header = ['Dataset',
                  'Criterion Name',
                  'Trial Number',

                  'Attribute Name (Monte Carlo framework)',
                  'Attribute Position (Monte Carlo framework)',
                  'Attribute p-value (Monte Carlo framework)',
                  'Attribute Name (Precise p-value)',
                  'Attribute Position (Precise p-value)',
                  'Attribute p-value (Precise p-value)',

                  'Are Attributes Different?',
                  'p-value Difference (between 0 and 1)']
        print(','.join(header), file=fout)

        for dataset_name in raw_data:
            for criterion_name in raw_data[dataset_name]:
                precise_criterion_name = ' '.join((criterion_name, 'Monte Carlo'))
                for (trial_number,
                     position_accepted) in sorted(raw_data[dataset_name][criterion_name].items()):

                    all_attributes_p_values = raw_p_value_data[
                        dataset_name][precise_criterion_name][trial_number]

                    if all_attributes_p_values:
                        (best_attrib_name,
                         best_attrib_p_value,
                         best_attrib_pos) = min(all_attributes_p_values.values(),
                                                key=lambda x: (x[1], x[2]))
                    else: # No valid attributes
                        (best_attrib_name,
                         best_attrib_p_value,
                         best_attrib_pos) = ('None', 'None', 'None')

                    if position_accepted != 'None':
                        try:
                            (accepted_attrib_name,
                             accepted_attrib_p_value,
                             _) = all_attributes_p_values[position_accepted]
                        except ValueError:
                            print('Missing position accepted:')
                            print('dataset_name =', dataset_name)
                            print('criterion_name =', criterion_name)
                            print('trial_number =', trial_number)
                            continue
                        are_different = best_attrib_pos != position_accepted
                        p_value_difference = accepted_attrib_p_value - best_attrib_p_value
                    else:
                        (accepted_attrib_name,
                         accepted_attrib_p_value) = ('None', 'None')
                        are_different = None
                        p_value_difference = None

                    print(','.join([dataset_name,
                                    criterion_name,
                                    str(trial_number),
                                    accepted_attrib_name,
                                    position_accepted,
                                    str(accepted_attrib_p_value),
                                    best_attrib_pos,
                                    best_attrib_name,
                                    str(best_attrib_p_value),
                                    str(are_different),
                                    str(p_value_difference)]),
                          file=fout)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please include a path to an experiment output folder.')
        sys.exit(1)

    main(sys.argv[1].replace(r'\ ', ' '))
