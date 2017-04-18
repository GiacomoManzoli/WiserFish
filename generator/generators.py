import datetime
import math
import random

from util import weighted_choice

from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


def generate_probabilites(y):
    # type: (np.ndarray) -> np.ndarray
    """Generate an array of probability value, one for each istance y.
    The value is influenced by the class of the istance"""
    class_cnt = np.bincount(y).size
    prob_interval = 1 / float(class_cnt)
    probs = np.zeros(shape=y.shape)

    for i in range(0, y.size):
        probs[i] = random.uniform(prob_interval * y[i], prob_interval * (y[i]+1))
    return probs


###########################
# CLIENTS DATA
###########################

def generate_clients(cnt):
    # type: (int) -> pd.DataFrame

    features = 10
    (X, y) = make_classification(n_samples=cnt, n_features=features, n_informative=4, n_repeated=2, n_classes=4,
                                 random_state=123)
    # X[i] array representing the i-th client, y[i] client's class (4 possible classes)
    pc = generate_probabilites(y)  # vector with order probabilities
    ids = [i for i in range(0, cnt)]
    merged = np.c_[ids, pc, X, y]
    names = np.array(["client_" + str(i) for i in range(0, cnt)])
    clients = pd.DataFrame(names, columns=["client_name"])

    col_names = ["clientId", "pc"] + ["client_feature_" + str(i) for i in range(0, features)] + ["class"]
    clients_data = pd.DataFrame(merged, columns=col_names)

    clients = clients.join(clients_data)
    return clients


###########################
# PRODUCTS DATA
###########################

def generate_products(cnt):
    # type: (int) -> pd.DataFrame

    features = 15
    (X, y) = make_classification(n_samples=cnt, n_features=features, n_informative=4, n_repeated=2, n_classes=4,
                                 random_state=456)
    # X[i] array representing the i-th product, y[i] product's class (4 possible classes)
    pp = generate_probabilites(y)
    ids = [i for i in range(0, cnt)]
    merged = np.c_[ids, pp, X, y]
    names = np.array(["prod_" + str(i) for i in range(0, cnt)])
    products = pd.DataFrame(names, columns=["product_name"])

    col_names = ["productId", "pp"] + ["product_feature_" + str(i) for i in range(0, features)] + ["class"]
    products_data = pd.DataFrame(merged, columns=col_names)

    products = products.join(products_data)
    return products


###########################
# PERIODIC DATA
###########################

def period_probability(t):
    # type: (int) -> float
    t %= 365
    prob = (math.sin(math.radians(t)) + 1) / 2  # converts t in radiants
    return prob


###########################
# ORDER DATA
###########################

def will_order(client, product, timestamp):
    # type: (pd.Series, pd.Series, int) -> int
    # client : array representing the client
    # product : array representing the product
    # timestamp : day of the order

    order_date = datetime.datetime.fromtimestamp(timestamp)
    # print order_date.strftime('%Y-%m-%d %H:%M:%S')
    t = order_date.timetuple().tm_yday  # Day of the year

    p_t = period_probability(t)
    p_c = client['pc']
    p_p = product['pp']
    p = p_t * p_p * p_c
    # print "--- Probabilities --- "
    # print "P_c =", p_c
    # print "P_p =", p_p
    # print "P_t =", p_t
    # print "Total =", p

    choices = [(1, p), (0, 1 - p)]
    return weighted_choice(choices)


def generate_matrix(clients, products, t):  # t --> timestamp
    clients_count = clients.shape[0]
    products_count = products.shape[0]
    matrix = np.ndarray(shape=(clients_count, products_count))
    print "Matrix size: ", matrix.shape
    # print "-----"

    for c in range(0, clients_count):
        for p in range(0, products_count):
            matrix[c, p] = will_order(clients.ix[c], products.ix[p], t)
    return matrix


def generate_orders(clients, products, days):
    # days : list of timestamp of the orders matrices that must be generated
    # returns the matrices of with the orders
    print "Generating orders:"
    print "#Clients:", clients.shape
    print "#Products:", products.shape
    matrices = {}
    for day in days:
        print "Generating orders for day: ", datetime.date.fromtimestamp(day).strftime('%Y-%m-%d %H:%M:%S')
        matrices[day] = generate_matrix(clients, products, day)
    return matrices
