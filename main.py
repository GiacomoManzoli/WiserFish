#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from sklearn import tree, svm
from sklearn.metrics import roc_auc_score
import pydotplus

from predictor.baseline import BaselinePredictor
from predictor.metrics import calculate_metrics
from predictor.proposizionalizer import proposizionalize
from generator.generators import generate_dataset, generate_orders

SECS_IN_DAY = 60*60*24


clients, products, orders, model = generate_dataset(clients_count=10,
                                                    products_count=10,
                                                    days_count=20,
                                                    day_interval=4,
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


# Predictions
clf = tree.DecisionTreeClassifier()
#clf = svm.SVC(kernel='rbf')
clf = clf.fit(X, y)
# print tree
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("tree_plot.pdf")

print "Today orders"
expected = generate_orders(clients, products, [today], model)[today]
print "--- Expected Result ---"
print expected

predictions_tree = clf.predict(queries_prop)
predictions_tree = predictions_tree.reshape(expected.shape)
print "--- Predicted (Tree) ---"
print predictions_tree

print "--- Predicted (Base) ---"
base_predictor = BaselinePredictor()
base_predictor.fit(orders)
predictions_base = base_predictor.predict_with_topn()
print predictions_base


######## METRICS
print "\n\n\n"
expected_vec = np.reshape(expected, expected.size)
predicted_tree_vec = clf.predict_proba(queries_prop)
predicted_base_vec = np.reshape(base_predictor.weights, base_predictor.weights.size)

#print expected_vec.shape, predicted_base_vec.shape

accuracy_tree, precision_tree, recall_tree = calculate_metrics(predictions_tree, expected)
# roc_auc_tree = roc_auc_score(y_true=expected_vec, y_score=predicted_tree_vec)
roc_auc_tree = -1

accuracy_base, precision_base, recall_base = calculate_metrics(predictions_base, expected)
roc_auc_base = roc_auc_score(y_true=expected_vec, y_score=predicted_base_vec)

print u"┌────────────┬──────────┬──────────┐"
print u"│  Metrics   │   Base   │   Tree   │"
print u"├────────────┼──────────┼──────────┤"
print u"│ Accuracy   │ ", unicode(u'%.4f  │  %.4f  │' % (accuracy_base, accuracy_tree))
print u"│ Precision  │ ", unicode(u'%.4f  │  %.4f  │' % (precision_base, precision_tree))
print u"│ Recall     │ ", unicode(u'%.4f  │  %.4f  │' % (recall_base, recall_tree))
print u"│ ROC_AUC    │ ", unicode(u'%.4f  │    --    │' % (roc_auc_base))
print u"└────────────┴──────────┴──────────┘"



