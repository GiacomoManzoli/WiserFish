import datetime
import math
import random
import time

from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

from generator.probability_models import RandomProbabilityModel, SimpleProbabilityModel, CondProbabilityModel

SECS_IN_DAY = 60*60*24


def generate_probabilites(y):
    # type: (np.array) -> np.array
    """Generate an array of probability value, one for each istance y.
    The value is influenced by the class of the istance"""
    class_cnt = np.bincount(y).size
    prob_interval = 1 / float(class_cnt)
    probs = np.array(y, dtype=np.float)

    for i in range(0, y.size):
        probs[i] = random.uniform(prob_interval * y[i], prob_interval * (y[i]+1))

    return probs


def generate_freq_scales(y):
    # type: (np.array) -> np.array
    """Generate an array of frequency scale value, one for each istance y.
    The value is influenced by the class of the istance"""
    scales = np.array(y, dtype=np.float)

    for i in range(0, y.size):
        c = scales[i]

        scale = 1.0
        if int(c) / 2 == 0:
            scale *= random.random()  # low frequency
        else:
            scale *= random.uniform(2, 365)  # high frequency

        if int(c) % 2 == 1:
            scale *= -1  # opposite frequency

        scales[i] = scale
    return scales


###########################
# CLIENTS DATA
###########################

def generate_clients(cnt):
    # type: (int) -> pd.DataFrame

    features = 10
    (X, y) = make_classification(n_samples=cnt, n_features=features, n_informative=4, n_repeated=2, n_classes=4,
                                 random_state=123)
    # X[i] array representing the i-th client, y[i] client's class (4 possible classes)

    # 4 different classes of order probabilities for the clients (used to determine `pc`)
    clients_order_probs = [random.randint(0, 3) for _ in range(0, cnt)]
    # 4 different classses of order frequency for the clients (used to determine `client_freq`)
    clients_order_freqs = [random.randint(0, 3) for _ in range(0, cnt)]

    # vector with clients' ids
    ids = [i for i in range(0, cnt)]
    # vector with clients' names
    names = ["client_" + str(i) for i in range(0, cnt)]

    # vector with order probabilities
    pc = generate_probabilites(np.array(clients_order_probs))
    # vector with order frequence scales
    freq_scale = generate_freq_scales(np.array(clients_order_freqs))

    # build the dataframe
    merged = np.c_[ids, pc, freq_scale, X, y]
    clients = pd.DataFrame(names, columns=["client_name"]) # the names can't be put inside `merged`

    col_names = ["clientId", "pc", "client_freq_scale"] + ["client_feature_" + str(i) for i in range(0, features)] + ["class"]
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

    # 4 different classes of order probabilities for the clients (used to determine `pc`)
    products_order_probs = [random.randint(0, 3) for _ in range(0, cnt)]
    # 4 different classses of order frequency for the clients (used to determine `client_freq`)
    products_order_freqs = [random.randint(0, 3) for _ in range(0, cnt)]

    # vector with products' ids
    ids = [i for i in range(0, cnt)]
    # vector with products' names
    names = ["prod_" + str(i) for i in range(0, cnt)]

    # vector with order probabilities
    pp = generate_probabilites(np.array(products_order_probs))
    # vector with order frequece scales
    freq_scale = generate_freq_scales(np.array(products_order_freqs))

    # build the dataframe
    merged = np.c_[ids, pp, freq_scale, X, y]
    products = pd.DataFrame(names, columns=["product_name"])

    col_names = ["productId", "pp", "product_freq_scale"] + ["product_feature_" + str(i) for i in range(0, features)] + ["class"]
    products_data = pd.DataFrame(merged, columns=col_names)

    products = products.join(products_data)

    return products


###########################
# ORDER DATA
###########################

def generate_matrix(clients, products, t, model):  # t --> timestamp
    clients_count = clients.shape[0]
    products_count = products.shape[0]
    matrix = np.ndarray(shape=(clients_count, products_count))

    for c in range(0, clients_count):
        for p in range(0, products_count):
            # model.probabi
            # client_copy = clients.ix[c].copy()
            # client_copy['clientId'] = c
            # product_copy = products.ix[p].copy()
            # product_copy['productId'] = p
            # matrix[c, p] = model.probability(client_copy, product_copy, t)
            matrix[c, p] = model.will_make_order(clients.ix[c], products.ix[p], t)
    return matrix


def generate_orders(clients, products, days, model):
    # days : list of timestamp of the orders matrices that must be generated
    # returns the matrices of with the orders
    print "Generating orders:"
    print "- # Clients:", clients.shape
    print "- # Products:", products.shape
    matrices = {}
    print ""
    generated = 0
    for day in days:
        print "Generating orders for day: ", datetime.date.fromtimestamp(day).strftime('%Y-%m-%d %H:%M:%S'), '[%d / %d]' % (generated+1, len(days))
        matrices[day] = generate_matrix(clients, products, day, model)
        generated += 1
    return matrices


###########################
# DATASET
###########################

def generate_days(days_count, day_interval=0):
    return [time.time() - SECS_IN_DAY * i * (day_interval + 1) for i in range(1, days_count+1)]  # starts from yesterday


def generate_dataset(clients_count, products_count, days_count, day_interval=0, model_name='random'):
    """

    :param clients_count: number of clients
    :param products_count: number of products
    :param days_count: number of days (matrices)
    :param day_interval: interval between each day (default=0, continuos matrices)
    :param model_name: probability model name {'random','simple', 'cond'}
    :return: clients, products, orders
    """
    print "Generating clients..."
    clients = generate_clients(clients_count)
    print "Generating products..."
    products = generate_products(products_count)

    days = generate_days(days_count, day_interval)

    print "Generating orders..."
    model = RandomProbabilityModel()
    if model_name == 'simple':
        model = SimpleProbabilityModel()
    elif model_name == 'cond':
        model = CondProbabilityModel(clients, products)

    if days_count != 0:
        orders = generate_orders(clients, products, days, model)
    else:
        orders = {}

    return clients, products, orders, model


###########################
# UTILITIES
###########################

def __check_generated_days(days):
    today_ts = time.time()
    str_time = '%Y-%m-%d'

    print "Today", datetime.datetime.fromtimestamp(today_ts).strftime(str_time)
    print "----"

    counters = {}

    for day in days:
        day_date = datetime.datetime.fromtimestamp(day)
        t = day_date.timetuple().tm_yday

        if t not in counters.keys():
            counters[t] = 0
        counters[t] += 1
        print "Day", day_date.strftime(str_time), "- T", day_date.timetuple().tm_yday

    print "----"
    for k in counters.keys():
        print "Period", k, "Frequence", counters[k]