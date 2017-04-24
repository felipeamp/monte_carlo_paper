#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Module used to run any experiment.
'''


import json
import os
import sys

import cross_validation_experiment
import train_and_test_experiment


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please include a path to an experiment configuration file.')
        sys.exit(1)

    experiment_config_filepath = sys.argv[1]
    if (not os.path.exists(experiment_config_filepath)
            or not os.path.isfile(experiment_config_filepath)):
        print('The path entered is NOT a valid experiment configuration file.')
        sys.exit(1)

    with open(experiment_config_filepath, 'r') as experiment_config_json:
        experiment_config = json.load(experiment_config_json)
    if experiment_config["use cross-validation"]:
        cross_validation_experiment.main(experiment_config)
    else:
        train_and_test_experiment.main(experiment_config)
