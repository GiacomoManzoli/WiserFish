#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getopt
import sys
import time

import numpy as np
import sqlite3

import pandas as pd
from sklearn.metrics import roc_auc_score

from dataset.generator.values import *
from dataset.manager import save_dataset, load_orders_dataframe
from predictor.baseline import BaselinePredictor
from predictor.less_sinful_baseline import LessSinfulBaselinePredictor
from predictor.multi_regressor_predictor import MultiRegressorPredictor
from predictor.single_regressor_predictor import SingleRegressorPredictor
from util import ConfigurationFile


def main(argv):
    load_from_file = False
    json_file_path = ""

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

    dataset_star_ts = config.starting_day
    dataset_end_ts = long(dataset_star_ts + (config.days_count - 1) * SECS_IN_DAY)
    train_set_start_ts = long(dataset_star_ts)
    train_set_end_ts = long(dataset_end_ts - 7 * SECS_IN_DAY)

    cnx = sqlite3.connect('./datasets/%s/data/%s.db' % (config.base_prefix, config.base_prefix))

    query_ts = train_set_end_ts + SECS_IN_DAY
    query_ts = 1493244000

    single = SingleRegressorPredictor(regressor_name='SVR')
    single.fit(cnx, clients_count=config.clients_count, products_count=config.products_count,
               from_ts=train_set_start_ts, to_ts=train_set_end_ts)
    pred_1 = single.predict_proba(query_ts)

    multi = MultiRegressorPredictor(components=['w_c', 'w_p', 'w_t', 'w_cp'], regressor_name='SVR')
    multi.fit(cnx, clients_count=config.clients_count, products_count=config.products_count,
              from_ts=train_set_start_ts, to_ts=train_set_end_ts)

    pred_multi = multi.predict_proba(query_ts)

    query = "select client_id, product_id " \
            "from orders " \
            "where datetime == %d " \
            "order by client_id, product_id" % query_ts
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

    base = BaselinePredictor()
    base.fit(cnx, clients_count=config.clients_count, products_count=config.products_count,
             from_ts=train_set_start_ts, to_ts=train_set_end_ts)
    pred_base = base.predict_proba(query_ts)

    less = LessSinfulBaselinePredictor()
    less.fit(cnx, clients_count=config.clients_count, products_count=config.products_count,
             from_ts=train_set_start_ts, to_ts=train_set_end_ts)
    pred_less = less.predict_proba(query_ts)

    test_vec_sum = np.sum(y_test)
    can_calculate_auc = test_vec_sum != 0 and test_vec_sum != y_test.size
    if can_calculate_auc:
        roc_auc_base = roc_auc_score(y_true=y_test, y_score=pred_base[:, 1])
        roc_auc_less = roc_auc_score(y_true=y_test, y_score=pred_less[:, 1])
        roc_auc_1 = roc_auc_score(y_true=y_test, y_score=pred_1[:, 1])
        roc_auc_multi = roc_auc_score(y_true=y_test, y_score=pred_multi[:, 1])
        print "AUC"
        print roc_auc_base, roc_auc_less, roc_auc_1, roc_auc_multi


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
