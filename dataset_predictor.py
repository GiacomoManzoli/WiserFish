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
import sqlite3
from sklearn import tree, ensemble
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import roc_auc_score

from dataset.generator.values import SECS_IN_DAY
from dataset.manager import load_orders_dataframe
from predictor.baseline import BaselinePredictor
from predictor.less_sinful_baseline import LessSinfulBaselinePredictor
from predictor.metrics import calculate_metrics
from predictor.multi_regressor_predictor import MultiRegressorPredictor
from predictor.single_regressor_predictor import SingleRegressorPredictor
from util import ConfigurationFile, Log

OUT_DIR_NAME = "outputs"
TAG = "dataset_predictor"


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

    cnx = sqlite3.connect('./datasets/%s/data/%s.db' % (config.base_prefix, config.base_prefix))

    dataset_star_ts = config.starting_day
    dataset_end_ts = long(dataset_star_ts + (config.days_count - 1) * SECS_IN_DAY)
    train_set_start_ts = long(dataset_star_ts)
    train_set_end_ts = long(dataset_end_ts - 7 * SECS_IN_DAY)

    Log.d(TAG, "Dataset loaded!")
    Log.d(TAG, "Total duration: " + str(time.time() - load_start_time) + " seconds")

    ############################
    # Test set generation
    ############################
    Log.d(TAG, "Genero le query...")

    query_ts = [
        train_set_end_ts + SECS_IN_DAY,  # Giorno immediatamente successivo alla fine del TS
        train_set_end_ts + 2 * SECS_IN_DAY,  # Due giorni dopo la fine del TS
        train_set_end_ts + 7 * SECS_IN_DAY  # Una settimana dopo la fine del TS
    ]

    queries = []
    for ts in query_ts:
        query = "select client_id, product_id " \
                "from orders " \
                "where datetime == %d " \
                "order by client_id, product_id" % ts
        # ^ ORDER BY è fondamentale per effettuare la creazione in modo efficiente
        cursor = cnx.execute(query)

        next_row = cursor.fetchone()
        df_rows = []
        for c in range(0, config.clients_count):
            for p in range(0, config.products_count):
                ordered = 0
                if next_row is not None and next_row == (c, p):
                    ordered = 1
                    next_row = cursor.fetchone()

                df_rows.append({
                    'ordered': ordered
                })
        y_test = pd.DataFrame(df_rows,
                              columns=['ordered'])['ordered'].as_matrix()

        queries.append(
            (datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d'),
             ts,
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
        # ("tree_10", tree.DecisionTreeClassifier(max_depth=10)),
        # ("tree_N", tree.DecisionTreeClassifier()),
        # ("bern", BernoulliNB()),
        # ("forest", ensemble.RandomForestClassifier()),
        ("svr_multi_pc_pp_pt_p_cp", MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t', 'w_cp'],
                                                            regressor_name='SVR')),
        ("svr_single", SingleRegressorPredictor(regressor_name='SVR')),

        ("par_multi_pc_pp", MultiRegressorPredictor(components=['w_c', 'w_p'],
                                                    regressor_name='PAR')),
        ("par_multi_pc_pp_pt", MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t'],
                                                       regressor_name='PAR')),
        ("par_multi_pc_pp_pt_p_cp", MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t', 'w_cp'],
                                                            regressor_name='PAR')),
        ("par_single", SingleRegressorPredictor(regressor_name='PAR'))
    ]

    with open("%s/%s/%s.csv" % (OUT_DIR_NAME, config.base_prefix, run_name), 'wb') as output:
        writer = csv.writer(output, delimiter=';')
        writer.writerow(['query', 'name', str('acc'), str('prec'), str('rec'), str('roc_auc')])
        for pair in clfs:
            name, clf = pair

            if run_name not in name and name != "base" and name != "less" and run_name != "all":
                continue

            print "--- Classifier:", name, "---"

            print "Fitting..."
            if "multi" in name or "single" in name or name == "base" or name == "less":
                clf.fit(cnx=cnx,
                        from_ts=train_set_start_ts,
                        to_ts=train_set_end_ts,
                        clients_count=config.clients_count,
                        products_count=config.products_count)
            else:
                #clf.fit(X_train, y_train)
                Log.d(TAG, "Non ancora implementato")

            for query in queries:
                query_name, ts, y_test = query

                Log.d(TAG, "Query: %s" % query_name)

                y_test_matrix = y_test.reshape((config.clients_count, config.products_count))
                Log.d(TAG,"Expected:")
                print y_test_matrix

                predictions = None
                predictions_probabilities = None

                if "multi" in name or "single" in name or name == "base" or name == "less":
                    predictions = clf.predict_with_topn(ts)  # reshape as a NClients x NProducts matrix
                    predictions_probabilities = clf.predict_proba(ts)
                else:
                    # predictions = clf.predict(X_test)
                    # predictions = predictions.reshape(
                    #     y_test_matrix.shape)  # reshape as a NClients x NProducts matrix
                    # predictions_probabilities = clf.predict_proba(X_test)
                    Log.d(TAG, "Non ancora implementato")

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

            del clf


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
