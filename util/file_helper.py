import json

import pandas as pd
import numpy as np
import pickle
import os


# Nomi delle directory
D_OUTPUTS = "./data"

def output_dir(prefix):
    experiment_path = "%s/%s/" % (D_OUTPUTS, prefix)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    return experiment_path


def save_all(clients, products, orders, model, name='', prefix=''):
    # type: (pd.DataFrame, pd.DataFrame, dict, str, str) -> None
    """Saves the pandas DataFrame and the orders dict as csv files inside the 'data' directory"""

    if name == '':
        name = prefix
    base_path = output_dir(name)

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    print "Saving into:", base_path, prefix
    clients.to_csv('%s%sclients.csv' % (base_path, prefix), index=False)
    products.to_csv('%s%sproducts.csv' % (base_path, prefix), index=False)
    with open('%s%spmodel.pkl' % (base_path, prefix), 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    # orders -> dictionary, keys:Datetime, values:np.ndarray
    # saves only the orders (the ones of the matrix)

    orders_rows = []
    for key in orders.keys():
        matrix = orders[key]
        clients_count = matrix.shape[0]
        products_count = matrix.shape[1]
        for c in range(0, clients_count):
            for p in range(0, products_count):
                if matrix[c, p] == 1:
                    orders_rows.append({
                        'datetime': key,  # timestamp of the key (order's date)
                        'clientId': clients.iloc[c]['clientId'],
                        'productId': products.iloc[p]['productId']
                    })
    if len(orders_rows) != 0:
        orders_df = pd.DataFrame(orders_rows, columns=['datetime', 'clientId', 'productId'])
        orders_df.to_csv('%s%sorders.csv' % (base_path, prefix), index=False)


def save_partial_orders(name, prefixes, result_prefix):
    assert len(prefixes) >= 1
    base_path = output_dir(name)

    print "Saving orders into:", base_path

    orders = None
    # Creates a single DF from all the partial files
    for prefix in prefixes:
        if prefix != '' and prefix[-1] != '_':
            prefix += '_'
        partial_df = pd.read_csv('%s%sorders.csv' % (base_path, prefix))
        if orders is None:
            orders = partial_df
        else:
            orders = pd.concat([orders, partial_df])

    orders.to_csv('%s%sorders.csv' % (base_path, result_prefix), index=False)


def load_all(name='', prefix=''):
    """Loads the clients and products DataFrames and the orderd dictionary from the 'data' directory"""

    if name == '':
        name = prefix
    base_path = output_dir(name)

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    print "Loading from:", base_path, prefix
    clients = pd.read_csv('%s%sclients.csv' % (base_path, prefix))
    products = pd.read_csv('%s%sproducts.csv' % (base_path, prefix))
    with open('%s%spmodel.pkl' % (base_path, prefix), 'rb') as input_file:
        model = pickle.load(input_file)

    try:
        orders = pd.read_csv('%s%sorders.csv' % (base_path, prefix))

        clients_count = clients.shape[0]  # shape[0] == row count
        products_count = products.shape[0]

        orders_dict = {}
        for idx, row in orders.iterrows():
            key = int(row['datetime'])  # timestamp dell'ordine
            if key not in orders_dict.keys():
                orders_dict[key] = np.zeros(shape=(clients_count, products_count))

            orders_dict[key][int(row['clientId']), int(row['productId'])] = 1
    except IOError:
        print "Couldn't load the orders"
        orders_dict = {}

    return clients, products, orders_dict, model

# JSON field names
J_PREFIX = "prefix"
J_CLIENTS_COUNT = "clients_count"
J_PRODUCTS_COUNT = "products_count"
J_DAYS_COUNT = "days_count"
J_DAY_INTERVAL = "day_interval"
J_MODEL_NAME = "model_name"
J_PART_SIZE = "part_size"


class ConfigurationFile(object):

    def __init__(self, json_file_path):
        print "Loading configuration from: ", json_file_path
        with open(json_file_path) as json_file:
            json_data = json.load(json_file)
        print json_data

        self.base_prefix = json_data[J_PREFIX]
        self.clients_count = json_data[J_CLIENTS_COUNT]
        self.products_count = json_data[J_PRODUCTS_COUNT]
        self.days_count = json_data[J_DAYS_COUNT]
        self.day_interval = json_data[J_DAY_INTERVAL]  # continuous
        self.model_name = json_data[J_MODEL_NAME]  # 'random'
        self.part_size = json_data[J_PART_SIZE]
