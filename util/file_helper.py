import pandas as pd
import numpy as np


def save_all(clients, products, orders):
    # type: (pd.DataFrame, pd.DataFrame, dict) -> None
    """Saves the pandas DataFrame and the orders dict as csv files inside the 'data' directory"""
    clients.to_csv('./data/clients.csv', index=False)
    products.to_csv('./data/products.csv', index=False)

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
    orders_df.to_csv('./data/orders.csv', index=False)


def load_all():
    """Loads the clients and products DataFrames and the orderd dictionary from the 'data' directory"""
    clients = pd.read_csv('./data/clients.csv')
    products = pd.read_csv('./data/products.csv')
    orders = pd.read_csv('./data/orders.csv')

    clients_count = clients.shape[0]  # shape[0] == row count
    products_count = products.shape[0]

    orders_dict = {}
    for idx, row in orders.iterrows():
        key = int(row['datetime'])  # timestamp dell'ordine
        if key not in orders_dict.keys():
            orders_dict[key] = np.zeros(shape=(clients_count, products_count))

        orders_dict[key][int(row['clientId']), int(row['productId'])] = 1

    return clients, products, orders_dict
