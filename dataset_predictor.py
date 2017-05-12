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
from predictor.metrics import calculate_metrics
from predictor.proposizionalizer import proposizionalize
from generator.generators import generate_dataset, generate_orders
from util import load_all, ConfigurationFile, os

SECS_IN_DAY = 60*60*24

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
    if not os.path.exists(OUT_DIR_NAME):
        os.makedirs(OUT_DIR_NAME)
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
            load_from_file = True
            run_name = arg

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

        clients, products, orders, model = load_all(config.base_prefix, prefix)
        print "Dataset loaded!"
        print "Total duration: ", time.time() - load_start_time, "seconds"
    else:
        clients, products, orders, model = generate_dataset(clients_count=10,
                                                            products_count=100,
                                                            days_count=10,
                                                            day_interval=0,
                                                            model_name='cond')

    ############################
    # Train set generation
    ############################

    print "Proposizionalization..."
    # Train data (the first matrix are the order for today)
    dataset = proposizionalize(orders, clients, products)
    X = dataset.drop('ordered', axis=1).as_matrix()
    y = dataset['ordered'].as_matrix()
    X_base = orders

    ############################
    # Test set generation
    ############################

    print "Generation of the test set for today and after tomorrow..."
    # Query data
    sample = orders[orders.keys()[0]]  # get the matrix dimension
    today_timestamp = time.time()  # current timestamp
    after_tomorrow = datetime.date.today() + datetime.timedelta(days=2)
    after_tomorrow_timestamp = time.mktime(after_tomorrow.timetuple())
    days = [today_timestamp, after_tomorrow_timestamp]

    # Genetares the proposizonalized dataset (x_test) for the query (no target value)
    query_data = {today_timestamp: np.zeros(shape=sample.shape)}
    query_data_prop = proposizionalize(query_data, clients, products)
    query_data_prop = query_data_prop.drop('ordered', axis=1)  # drops the target column

    # Generates the y_test for the dataset using the probability model
    expected = generate_orders(clients, products, days, model)

    expected_today = expected[today_timestamp]
    expected_today_vec = np.reshape(expected_today, expected_today.size)

    expected_after_tomorrow = expected[after_tomorrow_timestamp]
    expected_after_tomorrow_vec = np.reshape(expected_after_tomorrow, expected_after_tomorrow.size)

    all_queries = [
        ('today', query_data_prop, expected_today, expected_today_vec),
        ('after_tomorrow', query_data_prop, expected_after_tomorrow, expected_after_tomorrow_vec)
    ]

    ############################
    # Predictions
    ############################
    base_predictor = BaselinePredictor()
    tree5 = tree.DecisionTreeClassifier(max_depth=5)
    treeN = tree.DecisionTreeClassifier()
    tree10 = tree.DecisionTreeClassifier(max_depth=10)
    bern = BernoulliNB()
    forest = ensemble.RandomForestClassifier()

    clfs = [
        ("base", base_predictor),
        ("tree_5", tree5),
        ("tree_10", tree10),
        ("tree_N", treeN),
        ("bern", bern),
        ("forest", forest)
    ]

    with open("%s/%s.csv" % (OUT_DIR_NAME, run_name), 'wb') as output:
        writer = csv.writer(output, delimiter=';')
        writer.writerow(['query', 'name', str('acc'), str('prec'), str('rec'), str('roc_auc'), str('alt_roc_auc')])
        for query in all_queries:
            qname, X_test_prop, y_test, y_test_vec = query
            # Note: X_test_prop is a np.Dataframe and it can be used with SKlearn
            print "Expected:"
            print y_test
            for pair in clfs:
                name, clf = pair
                print "--- Classifier:", name, "---"
                if name == "base":
                    clf.fit(X_base)  # BaselinePredictor works in a different way then a sklearn classifier
                    predictions = clf.predict_with_topn()  # returns a NClients x NProducts matrix
                    predictions_probabilities = clf.predict_proba()  # retruns a matrix like a SKLearn classifier
                else:
                    clf.fit(X, y)
                    predictions = clf.predict(X_test_prop)
                    predictions = predictions.reshape(y_test.shape)  # reshape as a NClients x NProducts matrix
                    predictions_probabilities = clf.predict_proba(X_test_prop)

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
                writer.writerow([qname, name, str(acc), str(prec), str(rec), str(roc_auc), str(alt_roc_auc)])


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"