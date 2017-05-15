# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime


def proposizionalize(orders, clients, products):
    # type: (dict, pd.DataFrame, pd.DataFrame) -> pd.DataFrame
    """
    Effettua la proposizionalizzazione dei dati.
    :param orders: dizionario contente le matrici degli ordini da proposizionalizzare.
        Il dizionario è indicizzato per data (timestamp)
    :param clients: dataframe with clients' data
    :param products: dataframe with products' data
    :return: proposizionalized dataframe
    """

    # 1. Crea un data frame per gli ordini contenente anche le righe per i "non-ordini"
    orders_rows = []
    for key in orders.keys():
        matrix = orders[key]
        clients_count = matrix.shape[0]
        products_count = matrix.shape[1]
        for c in range(0, clients_count):
            for p in range(0, products_count):
                order_date = datetime.fromtimestamp(key)
                day_of_year = order_date.timetuple().tm_yday
                year = order_date.timetuple().tm_year
                orders_rows.append({
                    'datetime': key,  # timestamp della data dell'ordine
                    'day_of_year': day_of_year,
                    'year': year,
                    'clientId': clients.iloc[c]['clientId'],
                    'productId': products.iloc[p]['productId'],
                    'ordered': matrix[c, p]
                })

    orders_df = pd.DataFrame(orders_rows,
                             columns=['datetime', 'day_of_year', 'year', 'clientId', 'productId', 'ordered'])
    # Tipi corretti
    orders_df['datetime'] = orders_df['datetime'].astype(dtype=long)
    orders_df['day_of_year'] = orders_df['day_of_year'].astype(dtype=int)
    orders_df['year'] = orders_df['year'].astype(dtype=int)
    orders_df['clientId'] = orders_df['clientId'].astype(dtype=int)
    orders_df['productId'] = orders_df['productId'].astype(dtype=int)
    orders_df['ordered'] = orders_df['ordered'].astype(dtype=int)

    # 2. Effettua il join con gli altri dataframe
    orders_df = orders_df.join(clients, on='clientId', lsuffix='_o', rsuffix='_c')
    orders_df = orders_df.join(products, on='productId', lsuffix='_o', rsuffix='_p')

    # 2.1 Toglie le chiavi duplicate
    orders_df['client_id'] = orders_df['clientId_o']
    orders_df['product_id'] = orders_df['productId_o']
    orders_df = orders_df.drop(['clientId_o', 'clientId_c', 'productId_p', 'productId_o'], axis=1)

    # 3. Toglie i dati che sono stati usati per generare i dati
    orders_df = orders_df.drop('client_name', axis=1)
    orders_df = orders_df.drop('product_name', axis=1)

    orders_df = orders_df.drop('pc', axis=1)
    orders_df = orders_df.drop('client_freq_scale', axis=1)
    orders_df = orders_df.drop('pp', axis=1)
    orders_df = orders_df.drop('product_freq_scale', axis=1)

    return orders_df
