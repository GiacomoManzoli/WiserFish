#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import getopt, sys
import time

import datetime
import numpy as np
from sklearn import tree, svm, ensemble
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score

from predictor.baseline import BaselinePredictor
from predictor.less_sinful_baseline import LessSinfulBaselinePredictor
from predictor.metrics import calculate_metrics
from predictor.proposizionalizer import proposizionalize
from generator.generators import generate_dataset, generate_orders
from predictor.sinful_baseline import SinfulBaselinePredictor
from predictor.single_regressor_predictor import SingleRegressorPredictor
from util import ConfigurationFile, os, load_train_set, load_test_set


# JSON field names
J_PREFIX = "prefix"
J_CLIENTS_COUNT = "clients_count"
J_PRODUCTS_COUNT = "products_count"
J_DAYS_COUNT = "days_count"
J_DAY_INTERVAL = "day_interval"
J_MODEL_NAME = "model_name"
J_PART_SIZE = "part_size"

OUT_DIR_NAME = "outputs"


def main(argv):
    load_from_file = False
    json_file_path = ""
    run_name = "unamed_run"
    test_version = 0

    if not os.path.exists(OUT_DIR_NAME):
        os.makedirs(OUT_DIR_NAME)
    try:
        (opts, args) = getopt.getopt(argv, "f:v:r:")
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
        if opt in ("-v", "--testversion"):
            test_version = int(arg)

    if load_from_file:
        config = ConfigurationFile(json_file_path)

        # places it in the base_prefix data subdir.
        prefix = '%s_c%d_p%d_d%d_int%d_pmod_%s_' % (config.base_prefix,
                                                    config.clients_count,
                                                    config.products_count,
                                                    config.days_count,
                                                    config.day_interval,
                                                    config.model_name)
        print prefix
        load_start_time = time.time()

        clients, products, orders, model = load_train_set(config.base_prefix, prefix)
        _, _, test_orders, _ = load_test_set(config.base_prefix, prefix, test_version)
        print "Dataset loaded!"
        print "Total duration: ", time.time() - load_start_time, "seconds"
    else:
        clients, products, orders, model = generate_dataset(clients_count=10,
                                                            products_count=100,
                                                            days_count=10,
                                                            day_interval=0,
                                                            model_name='cond')
        print "Generation of the test set for today and after tomorrow..."
        # Query data
        today_timestamp = time.time()  # current timestamp
        after_tomorrow = datetime.date.today() + datetime.timedelta(days=2)
        after_tomorrow_timestamp = time.mktime(after_tomorrow.timetuple())
        days = [today_timestamp, after_tomorrow_timestamp]
        # Generates the y_test for the dataset using the probability model
        test_orders = generate_orders(clients, products, days, model)

    ############################
    # Train set generation
    ############################

    print "Proposizionalization..."
    # Train data (the first matrix are the order for today)
    dataset = proposizionalize(orders, clients, products)
    X = dataset.drop('ordered', axis=1).as_matrix()
    y = dataset['ordered'].as_matrix()
    X_base = orders

    if len(X_base.keys()) == 0:
        print "ERROR: EMPTY TRAIN SET"
        return

    ############################
    # Test set generation
    ############################
    print "Query generation..."
    matrix_shape = (clients.shape[0], products.shape[0]) # get the matrix dimension
    days = test_orders.keys()  # <- days[0] = today, days[1] = today+2
    queries = []
    for day_timestamp in days:
        # Generates the proposizonalized dataset (x_test) for the query (no target value)
        query_data = {day_timestamp: np.zeros(shape=matrix_shape)}
        # ^ creates an all zeros matrix for the orders
        #   The timestamp needs to be coherent with the query.
        query_data_prop = proposizionalize(query_data, clients, products)
        # ^ 1. creates a dataframe for the orders, all with the class "not ordered
        #   2. joins it with the clients and products dataframe to obtain the proposizionalized version
        #      The results contains N_clients x N_products rows, with an irrelevent `ordered` column
        query_data_prop = query_data_prop.drop('ordered', axis=1)  # drops the target column

        # Generates the ground truth (y_test) for x_test, divided into
        expected = test_orders[day_timestamp]
        expected_vec = np.reshape(expected, expected.size)
        queries.append(
            (datetime.datetime.fromtimestamp(day_timestamp).strftime('%Y-%m-%d'),
             query_data_prop,
             expected,
             expected_vec)
        )

    if len(queries) == 0:
        print "ERROR: EMPTY TEST SET"
        return
    ############################
    # Predictions
    ############################
    treeN = tree.DecisionTreeClassifier()
    tree10 = tree.DecisionTreeClassifier(max_depth=10)
    bern = BernoulliNB()
    forest = ensemble.RandomForestClassifier()

    clfs = [
        ("base", BaselinePredictor()),
        ("sinful", SinfulBaselinePredictor()),
        ("less", LessSinfulBaselinePredictor()),
        ("tree_5", tree.DecisionTreeClassifier(max_depth=5)),
        #("tree_10", tree10),
        #("tree_N", treeN),
        #("bern", bern),
        #("forest", forest),
        ("single_1", SingleRegressorPredictor(group_size=1))
    ]

    with open("%s/%s.csv" % (OUT_DIR_NAME, run_name), 'wb') as output:
        writer = csv.writer(output, delimiter=';')
        writer.writerow(['query', 'name', str('acc'), str('prec'), str('rec'), str('roc_auc'), str('alt_roc_auc')])
        for query in queries:
            query_name, X_test_prop, y_test, y_test_vec = query
            # Note: X_test_prop is a np.Dataframe and it can be used with SKlearn
            print "Expected:"
            print y_test
            for pair in clfs:
                name, clf = pair
                print "--- Classifier:", name, "---"
                # X or X_base (or orders) -> train data set
                # X_test_prop -> test data set (not needed for BasePredictor)
                predictions = None
                predictions_probabilities = None

                t = time.mktime(datetime.datetime.strptime(query_name, "%Y-%m-%d").timetuple())

                if len(X_base.keys()) > 0:
                    if name == "base":
                        clf.fit(orders)  # BaselinePredictor works in a different way then a sklearn classifier
                        predictions = clf.predict_with_topn(t)  # returns a NClients x NProducts matrix
                        predictions_probabilities = clf.predict_proba(t)  # retruns a matrix like a SKLearn classifier
                    elif name == "less":
                        clf.fit(orders)
                        predictions = clf.predict_with_topn(t)  # returns a NClients x NProducts matrix
                        predictions_probabilities = clf.predict_proba(t)  # retruns a matrix like a SKLearn classifier
                    elif name == "sinful":
                        clf.fit(clients, products, orders)
                        predictions = clf.predict_with_topn(t)  # returns a NClients x NProducts matrix
                        predictions_probabilities = clf.predict_proba(t)  # retruns a matrix like a SKLearn classifier
                    elif name == "single_1" or name == "single_3" or name == "single_5" or name == "single":
                        clf.fit(clients, products, orders)
                        predictions = clf.predict_with_topn(t)
                        predictions = predictions.reshape(y_test.shape)  # reshape as a NClients x NProducts matrix
                        predictions_probabilities = clf.predict_proba(t)
                    else:
                        clf.fit(X, y)
                        predictions = clf.predict(X_test_prop)
                        predictions = predictions.reshape(y_test.shape)  # reshape as a NClients x NProducts matrix
                        predictions_probabilities = clf.predict_proba(X_test_prop)

                if predictions is not None:
                    print predictions
                    acc, prec, rec = calculate_metrics(predictions, y_test)  # works on matrices

                    roc_auc = -1
                    alt_roc_auc = -1
                    test_vec_sum = np.sum(y_test_vec)
                    can_calculate_AUC = test_vec_sum != 0 and test_vec_sum != y_test_vec.size
                    if can_calculate_AUC:
                        # check to avoid that all the probabilities are 1 for the class 0
                        roc_auc = roc_auc_score(y_true=y_test_vec, y_score=predictions_probabilities[:, 1])
                        alt_roc_auc = roc_auc_score(y_true=y_test_vec, y_score=predictions_probabilities[:, 0])

                    print name, acc, prec, rec, roc_auc
                    writer.writerow([query_name, name, str(acc), str(prec), str(rec), str(roc_auc), str(alt_roc_auc)])


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"