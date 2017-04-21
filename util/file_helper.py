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


def save_all(clients, products, orders, model,name='', prefix=''):
    # type: (pd.DataFrame, pd.DataFrame, dict) -> None
    """Saves the pandas DataFrame and the orders dict as csv files inside the 'data' directory"""

    if name == '':
        name = prefix
    base_path = output_dir(name)

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    print base_path, prefix
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
    orders_df = pd.DataFrame(orders_rows, columns=['datetime', 'clientId', 'productId'])
    orders_df.to_csv('%s%sorders.csv' % (base_path, prefix), index=False)


def load_all(prefix=''):
    """Loads the clients and products DataFrames and the orderd dictionary from the 'data' directory"""

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    base_path = output_dir(prefix)
    clients = pd.read_csv('%s%sclients.csv' % (base_path, prefix))
    products = pd.read_csv('%s%sproducts.csv' % (base_path, prefix))
    orders = pd.read_csv('%s%sorders.csv' % (base_path, prefix))
    with open('%s%spmodel.pkl', 'rb') as input_file:
        model = pickle.load(input_file)

    clients_count = clients.shape[0]  # shape[0] == row count
    products_count = products.shape[0]

    orders_dict = {}
    for idx, row in orders.iterrows():
        key = int(row['datetime'])  # timestamp dell'ordine
        if key not in orders_dict.keys():
            orders_dict[key] = np.zeros(shape=(clients_count, products_count))

        orders_dict[key][int(row['clientId']), int(row['productId'])] = 1

    return clients, products, orders_dict, model
