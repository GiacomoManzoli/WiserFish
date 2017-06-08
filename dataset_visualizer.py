#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import datetime
import getopt
import os
import sys
import time

import math
import numpy as np
import pandas as pd
from sklearn import tree, ensemble
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB

from dataset.generator.generator_v2 import generate_dataset
from dataset.generator.values import SECS_IN_DAY
from dataset.manager import load_dataset, make_order_matrices
from dataset.proposizionalizer import proposizionalize
from dataset.set_maker import make_train_set, make_test_set
from predictor.baseline import BaselinePredictor
from predictor.less_sinful_baseline import LessSinfulBaselinePredictor
from predictor.metrics import calculate_metrics
from predictor.multi_regressor_predictor import MultiRegressorPredictor
from predictor.sinful_baseline import SinfulBaselinePredictor
from predictor.single_regressor_predictor import SingleRegressorPredictor
from util import ConfigurationFile

OUT_DIR_NAME = "outputs"


def main(argv):
    load_from_file = False
    json_file_path = ""

    if not os.path.exists(OUT_DIR_NAME):
        os.makedirs(OUT_DIR_NAME)
    try:
        (opts, args) = getopt.getopt(argv, "f:")
    except getopt.GetoptError, e:
        print "Wrong options"
        print e
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            load_from_file = True
            json_file_path = arg
    if not load_from_file:
        print "Errore: non è stato specificato il file!"
        return

    config = ConfigurationFile(json_file_path)

    load_start_time = time.time()

    # Dataframe contenti i clienti, prodotti e ordini
    clients_df, products_df, orders_df = load_dataset(config.base_prefix)

    print "Dataset loaded!"
    print "Total duration: ", time.time() - load_start_time, "seconds"

    dataset_star_ts = config.starting_day
    dataset_end_ts = long(dataset_star_ts + (config.days_count - 1) * SECS_IN_DAY)

    print "Dataset:", config.base_prefix
    print "Numero clienti:", config.clients_count
    print "Numero prodotti:", config.products_count
    print "--- Distribuzione clienti ---"
    print "Percentuale clienti GDO", config.gdo_ratio
    print "Percentuale clienti ristoranti", config.resturant_ratio
    print "Percentuale clienti negozi", 1-config.gdo_ratio - config.resturant_ratio
    print "--- Distribuzione tipologia clienti ---"
    print "Percentuale clienti a consumo", config.consumption_ratio
    print "Percentuale clienti regolari", config.regular_ratio
    print "Percentuale clienti casuali", 1 - config.regular_ratio - config.consumption_ratio
    print "--- Distribuzione prodotti ---"
    print "Prodotti stagionali", config.seasonal_ratio


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
