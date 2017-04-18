import time

import numpy as np
from sklearn import tree, svm
import pydotplus

from predictor import baseline_predictor
from predictor import calculate_metrics
from predictor.proposizionalizer import proposizionalize
from generator.generators import generate_orders, generate_clients, generate_products

SECS_IN_DAY = 60*60*24

print "Generating clients..."
clients = generate_clients(5)
# print clients

print "Generating products..."
products = generate_products(10)

cnt_days = 10
days = [time.time() - SECS_IN_DAY * i for i in range(1, cnt_days+1)]  # from yesterday to cnt_days back

orders = generate_orders(clients, products, days)


# Train data
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
expected = generate_orders(clients, products, [today])[today]
print "--- Expected Result ---"
print expected

predictions = clf.predict(queries_prop)
predictions = predictions.reshape(expected.shape)
print "--- Predicted (Tree) ---"
print predictions

print "--- Predicted (Base) ---"
predicted = baseline_predictor(orders, 0.0004)
print predicted

accuracy_tree, precision_tree, recall_tree = calculate_metrics(predictions, expected)
print "-- Tree metrics --"
print "Accuracy:", accuracy_tree
print "Precision:", precision_tree
print "Recall:", recall_tree

accuracy_base, precision_base, recall_base = calculate_metrics(predicted, expected)
print "-- Base metrics --"
print "Accuracy:", accuracy_base
print "Precision:", precision_base
print "Recall:", recall_base