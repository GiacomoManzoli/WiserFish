import pandas as pd
import numpy as np
import pickle


def save_all(clients, products, orders, model, prefix=''):
    # type: (pd.DataFrame, pd.DataFrame, dict) -> None
    """Saves the pandas DataFrame and the orders dict as csv files inside the 'data' directory"""

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    clients.to_csv('./data/%sclients.csv' % prefix, index=False)
    products.to_csv('./data/%sproducts.csv' % prefix, index=False)
    with open('./data/%spmodel.pkl' % prefix, 'wb') as output:
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
    orders_df.to_csv('./data/%sorders.csv' % prefix, index=False)


def load_all(prefix=''):
    """Loads the clients and products DataFrames and the orderd dictionary from the 'data' directory"""

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    clients = pd.read_csv('./data/%sclients.csv' % prefix)
    products = pd.read_csv('./data/%sproducts.csv' % prefix)
    orders = pd.read_csv('./data/%sorders.csv' % prefix)
    with open('./data/%spmodel.pkl', 'rb') as input:
        model = pickle.load(input)

    clients_count = clients.shape[0]  # shape[0] == row count
    products_count = products.shape[0]

    orders_dict = {}
    for idx, row in orders.iterrows():
        key = int(row['datetime'])  # timestamp dell'ordine
        if key not in orders_dict.keys():
            orders_dict[key] = np.zeros(shape=(clients_count, products_count))

        orders_dict[key][int(row['clientId']), int(row['productId'])] = 1

    return clients, products, orders_dict, model
