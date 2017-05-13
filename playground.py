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
from util import ConfigurationFile, os, load_train_set, load_test_set


def main(argv):
    clients, products, orders, model = generate_dataset(clients_count=10,
                                                        products_count=50,
                                                        days_count=50,
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
    base_predictor = BaselinePredictor()
    sinful = SinfulBaselinePredictor()
    less = LessSinfulBaselinePredictor()
    treeN = tree.DecisionTreeClassifier()

    clfs = [
        ("base", base_predictor),
        ("sinful", sinful),
        ("less", less),
        ("tree_N", treeN),
    ]

    for query in queries:
        query_name, X_test_prop, y_test, y_test_vec = query
        # Note: X_test_prop is a np.Dataframe and it can be used with SKlearn
        print "Expected:"
        print y_test
        for pair in clfs:
            name, clf = pair
            print "--- Classifier:", name, "---"
            # X or X_base -> train data set
            # X_test_prop -> test data set (not needed for BasePredictor)
            predictions = None
            predictions_probabilities = None
            if len(X_base.keys()) > 0:
                if name == "base":
                    clf.fit(X_base)  # BaselinePredictor works in a different way then a sklearn classifier
                    predictions = clf.predict_with_topn()  # returns a NClients x NProducts matrix
                    predictions_probabilities = clf.predict_proba()  # retruns a matrix like a SKLearn classifier
                elif name == "less":
                    t = time.mktime(datetime.datetime.strptime(query_name, "%Y-%m-%d").timetuple())
                    clf.fit(X_base)
                    predictions = clf.predict_with_topn(t)  # returns a NClients x NProducts matrix
                    predictions_probabilities = clf.predict_proba(t)  # retruns a matrix like a SKLearn classifier
                elif name == "sinful":
                    t = time.mktime(datetime.datetime.strptime(query_name, "%Y-%m-%d").timetuple())
                    clf.fit(clients, products, X_base)
                    predictions = clf.predict_with_topn(t)  # returns a NClients x NProducts matrix
                    predictions_probabilities = clf.predict_proba(t)  # retruns a matrix like a SKLearn classifier
                else:
                    clf.fit(X, y)
                    predictions = clf.predict(X_test_prop)
                    predictions = predictions.reshape(y_test.shape)  # reshape as a NClients x NProducts matrix
                    predictions_probabilities = clf.predict_proba(X_test_prop)

            if predictions is not None:
                # print predictions
                acc, prec, rec = calculate_metrics(predictions, y_test)  # works on matrices

                roc_auc = -1
                test_vec_sum = np.sum(y_test_vec)
                can_calculate_AUC = test_vec_sum != 0 and test_vec_sum != y_test_vec.size
                if can_calculate_AUC:
                    # check to avoid that all the probabilities are 1 for the class 0
                    roc_auc = roc_auc_score(y_true=y_test_vec, y_score=predictions_probabilities[:, 1])

                print name, acc, prec, rec, roc_auc


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"