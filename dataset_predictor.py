#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getopt, sys
import time
import numpy as np
from sklearn import tree, svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score

from predictor.baseline import BaselinePredictor
from predictor.metrics import calculate_metrics
from predictor.proposizionalizer import proposizionalize
from generator.generators import generate_dataset, generate_orders
from util import load_all, ConfigurationFile

SECS_IN_DAY = 60*60*24
# JSON field names
J_PREFIX = "prefix"
J_CLIENTS_COUNT = "clients_count"
J_PRODUCTS_COUNT = "products_count"
J_DAYS_COUNT = "days_count"
J_DAY_INTERVAL = "day_interval"
J_MODEL_NAME = "model_name"
J_PART_SIZE = "part_size"


def main(argv):
    load_from_file = False
    json_file_path = ""

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
        laod_start_cpu_time = time.clock()
        clients, products, orders, model = load_all(config.base_prefix, prefix)
        print "Dataset loaded!"
        print "Total duration: ", time.time() - load_start_time, "seconds"
        print "CPU time:", time.clock() - load_start_time, "seconds"
    else:
        clients, products, orders, model = generate_dataset(clients_count=5,
                                                            products_count=5,
                                                            days_count=5,
                                                            day_interval=0,
                                                            model_name='cond')

    # Train data (the first matrix are the order for today)
    dataset = proposizionalize(orders, clients, products)
    X = dataset.drop('ordered', axis=1).as_matrix()
    y = dataset['ordered'].as_matrix()

    # Query data
    sample = orders[orders.keys()[0]]  # get the matrix dimension
    queries = {}
    today = time.time()
    queries[today] = np.zeros(shape=sample.shape)  # empty matrix to reuse the proposizionalization function
    queries_prop = proposizionalize(queries, clients, products)
    queries_prop = queries_prop.drop('ordered', axis=1)  # drops the target column

    ############################
    # Predictions
    ############################

    print "Today orders"
    expected = generate_orders(clients, products, [today], model)[today]
    expected_vec = np.reshape(expected, expected.size)
    print "--- Expected Result ---"
    print expected

    # Tree

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

    predictions_tree = clf.predict(queries_prop)
    predictions_tree = predictions_tree.reshape(expected.shape)
    print "--- Predicted (Tree) ---"
    print predictions_tree

    # BernoulliNB

    clf_bern = BernoulliNB()
    clf_bern.fit(X, y)
    predictions_bern = clf_bern.predict(queries_prop)
    predictions_bern = predictions_bern.reshape(expected.shape)
    print "--- Predicted (Bern) ---"
    print predictions_bern

    predicted_bern_vec = clf_bern.predict_proba(queries_prop)
    predicted_bern_vec = predicted_bern_vec[:, 1]  # probabilities for class "1"
    accuracy_bern, precision_bern, recall_bern = calculate_metrics(predictions_bern, expected)
    roc_auc_bern = roc_auc_score(y_true=expected_vec, y_score=predicted_bern_vec)

    # LinearSVM

    #clf_lsvm = svm.SVC(kernel='rbf') # linear doesn't work (endless train), poly gives a "nan" decision_function
    #clf_lsvm.fit(X, y)
    #predictions_lsvm = clf_lsvm.predict(queries_prop)
    #predictions_lsvm = predictions_lsvm.reshape(expected.shape)
    #print "--- Predicted (LSVM) ---"
    #print predictions_lsvm
#
    #predicted_lsvm_vec = clf_lsvm.decision_function(queries_prop)
    #print predicted_lsvm_vec
    ##predicted_lsvm_vec = predicted_lsvm_vec[:, 1]  # probabilities for class "1"
    #accuracy_lsvm, precision_lsvm, recall_lsvm = calculate_metrics(predictions_lsvm, expected)
    #roc_auc_lsvm = roc_auc_score(y_true=expected_vec, y_score=predicted_bern_vec)

    # Baseline (TOPN)

    print "--- Predicted (Base) ---"
    base_predictor = BaselinePredictor()
    base_predictor.fit(orders)
    predictions_base = base_predictor.predict_with_topn()
    print predictions_base


    ######## METRICS
    print "\n\n\n"
    predicted_tree_vec = clf.predict_proba(queries_prop)
    #print predicted_tree_vec
    predicted_base_vec = np.reshape(base_predictor.weights, base_predictor.weights.size)
    #print expected_vec.shape, predicted_base_vec.shape
    accuracy_tree, precision_tree, recall_tree = calculate_metrics(predictions_tree, expected)
    # roc_auc_tree = roc_auc_score(y_true=expected_vec, y_score=predicted_tree_vec)
    roc_auc_tree = -1

    accuracy_base, precision_base, recall_base = calculate_metrics(predictions_base, expected)
    roc_auc_base = roc_auc_score(y_true=expected_vec, y_score=predicted_base_vec)

    print u"┌────────────┬──────────┬──────────┬──────────┐"
    print u"│  Metrics   │   Base   │   Tree   │   Bern   │"
    print u"├────────────┼──────────┼──────────┼──────────┤"
    print u"│ Accuracy   │ ", unicode(u'%.4f  │  %.4f  │  %.4f  │' % (accuracy_base, accuracy_tree, accuracy_bern))
    print u"│ Precision  │ ", unicode(u'%.4f  │  %.4f  │  %.4f  │' % (precision_base, precision_tree, precision_bern))
    print u"│ Recall     │ ", unicode(u'%.4f  │  %.4f  │  %.4f  │' % (recall_base, recall_tree, recall_bern))
    print u"│ ROC_AUC    │ ", unicode(u'%.4f  │    --    │  %.4f  │' % (roc_auc_base, roc_auc_bern))
    print u"└────────────┴──────────┴──────────┴──────────┘"

if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
