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
    run_name = "unamed_run"

    try:
        (opts, args) = getopt.getopt(argv, "f:r:")
    except getopt.GetoptError, e:
        print "Wrong options"
        print e
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            load_from_file = True
            json_file_path = arg
        if opt in ("-r", "--runname"):
            run_name = arg

    if not load_from_file:
        print "Errore: non è stato specificato il file!"
        return

    config = ConfigurationFile(json_file_path)

    if not os.path.exists(OUT_DIR_NAME):
        os.makedirs(OUT_DIR_NAME)
    if not os.path.exists(OUT_DIR_NAME + "/" + config.base_prefix):
        os.makedirs(OUT_DIR_NAME + "/" + config.base_prefix)

    load_start_time = time.time()

    # Dataframe contenti i clienti, prodotti e ordini
    clients_df, products_df, orders_df = load_dataset(config.base_prefix)

    print "Dataset loaded!"
    print "Total duration: ", time.time() - load_start_time, "seconds"

    dataset_star_ts = config.starting_day
    dataset_end_ts = long(dataset_star_ts + (config.days_count - 1) * SECS_IN_DAY)

    print orders_df.head(5)

    ############################
    # Train set generation
    ############################

    train_set_start_ts = long(dataset_star_ts)
    # train_set_end_ts = long(train_set_start_ts + (int(math.floor(config.days_count / 2)-1) * SECS_IN_DAY ))
    train_set_end_ts = long(dataset_end_ts - 7 * SECS_IN_DAY)
    # ^ il -1 serve perché gli estremi dell'intervallo sono entrambi compresi e quindi senza il -1 si avrebbe un
    # train set con days_count+1 giorni
    X_train_dict, X_train, y_train = make_train_set(clients_df=clients_df,
                                                    products_df=products_df,
                                                    orders_df=orders_df,
                                                    from_ts=train_set_start_ts,
                                                    to_ts=train_set_end_ts)

    if len(X_train_dict.keys()) == 0:
        print "ERROR: EMPTY TRAIN SET"
        return

    ############################
    # Test set generation
    ############################
    print "Query generation..."

    query_ts = [
        train_set_end_ts + SECS_IN_DAY,  # Giorno immediatamente successivo alla fine del TS
        train_set_end_ts + 2 * SECS_IN_DAY,  # Due giorni dopo la fine del TS
        train_set_end_ts + 7 * SECS_IN_DAY  # Una settimana dopo la fine del TS
    ]

    queries = []
    for ts in query_ts:
        X_test_dict, X_test, y_test = make_test_set(clients_df=clients_df,
                                                    products_df=products_df,
                                                    orders_df=orders_df,
                                                    query_ts=ts)

        queries.append(
            (datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d'),
             X_test_dict,
             X_test,
             y_test)
        )

    if len(queries) == 0:
        print "ERROR: EMPTY TEST SET"
        return

    ############################
    # Predictions
    ############################

    clfs = [
        ("base", BaselinePredictor()),
        # ("sinful", SinfulBaselinePredictor()), Non può più essere utilizzato, perché usa p_c e p_p
        ("less", LessSinfulBaselinePredictor()),
        # ("tree_5", tree.DecisionTreeClassifier(max_depth=5)),
        ("tree_10", tree.DecisionTreeClassifier(max_depth=10)),
        # ("tree_N", tree.DecisionTreeClassifier()),
        # ("bern", BernoulliNB()),
        # ("forest", ensemble.RandomForestClassifier()),
        ("svr_multi_pc_pp", MultiRegressorPredictor(components=['w_c', 'w_p'],
                                                    regressor_name='SVR')),
        ("svr_multi_pc_pp_pt", MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t'],
                                                       regressor_name='SVR')),
        ("svr_multi_pc_pp_pt_p_cp", MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t', 'w_cp'],
                                                            regressor_name='SVR')),
        ("svr_multi_pt_p_cp", MultiRegressorPredictor(components=['w_t', 'w_cp'],
                                                      regressor_name='SVR')),
        ('svr_single_1', SingleRegressorPredictor(group_size=1,
                                                  regressor_name='SVR')),
        # ('single_3', SingleRegressorPredictor(regressor_name='SVR', group_size=3)),
        # ('single_5', SingleRegressorPredictor(regressor_name='SVR', group_size=5)),
        ("sgd_multi_pc_pp", MultiRegressorPredictor(components=['w_c', 'w_p'],
                                                    regressor_name='SGD')),
        ("sgd_multi_pc_pp_pt", MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t'],
                                                       regressor_name='SGD')),
        ("sgd_multi_pc_pp_pt_p_cp", MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t', 'w_cp'],
                                                            regressor_name='SGD')),
        ("sgd_multi_pt_p_cp", MultiRegressorPredictor(components=['w_t', 'w_cp'],
                                                      regressor_name='SGD')),
        ('sgd_single_1', SingleRegressorPredictor(group_size=1,
                                                  regressor_name='SGD')),

        ("par_multi_pc_pp", MultiRegressorPredictor(components=['w_c', 'w_p'],
                                                    regressor_name='PAR')),
        ("par_multi_pc_pp_pt", MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t'],
                                                       regressor_name='PAR')),
        ("par_multi_pc_pp_pt_p_cp", MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t', 'w_cp'],
                                                            regressor_name='PAR')),
        ("par_multi_pt_p_cp", MultiRegressorPredictor(components=['w_t', 'w_cp'],
                                                      regressor_name='PAR')),
        ('par_single_1', SingleRegressorPredictor(group_size=1,
                                                  regressor_name='PAR'))
    ]

    with open("%s/%s/%s.csv" % (OUT_DIR_NAME, config.base_prefix, run_name), 'wb') as output:
        writer = csv.writer(output, delimiter=';')
        writer.writerow(['query', 'name', str('acc'), str('prec'), str('rec'), str('roc_auc')])
        for pair in clfs:
            name, clf = pair

            if run_name not in name and name != "base" and name != "less" and name!="all":
                continue

            print "--- Classifier:", name, "---"

            print "Fitting..."
            if name == "base" or name == "less":
                clf.fit(X_train_dict)
            elif "multi" in name or "single" in name:
                clf.fit(clients_df, products_df, X_train_dict)
            else:
                clf.fit(X_train, y_train)

            for query in queries:
                query_name, X_test_dict, X_test, y_test = query

                # X_test_dict: dati per il test in formato dizionario di matrice
                # X_test: dati per il test sotto forma di ndarray
                # y_test: label per i dati di test sotto forma di ndarray (vettore singolo)

                print "Query:", query_name

                y_test_matrix = y_test.reshape((config.clients_count, config.products_count))
                print "Expected:"
                print y_test_matrix

                predictions = None
                predictions_probabilities = None

                t = time.mktime(datetime.datetime.strptime(query_name, "%Y-%m-%d").timetuple())

                if len(X_test_dict.keys()) > 0:
                    if name == "base" or name == "less":
                        predictions = clf.predict_with_topn(t)  # returns a NClients x NProducts matrix
                        predictions_probabilities = clf.predict_proba(t)  # returns a matrix like a SKLearn classifier
                    elif "multi" in name or "single" in name:
                        predictions = clf.predict_with_topn(t)  # reshape as a NClients x NProducts matrix
                        predictions_probabilities = clf.predict_proba(t)
                    else:
                        predictions = clf.predict(X_test)
                        predictions = predictions.reshape(y_test_matrix.shape)  # reshape as a NClients x NProducts matrix
                        predictions_probabilities = clf.predict_proba(X_test)

                if predictions is not None:
                    print predictions
                    acc, prec, rec = calculate_metrics(predictions, y_test_matrix)  # funziona comparando le matrici

                    roc_auc = -1
                    test_vec_sum = np.sum(y_test)
                    can_calculate_AUC = test_vec_sum != 0 and test_vec_sum != y_test.size
                    if can_calculate_AUC and predictions_probabilities is not None:
                        roc_auc = roc_auc_score(y_true=y_test, y_score=predictions_probabilities[:, 1])

                    print name, acc, prec, rec, roc_auc
                    writer.writerow([query_name, name, str(acc), str(prec), str(rec), str(roc_auc)])



if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
