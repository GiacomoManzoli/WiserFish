# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from dataset.manager import make_order_matrices
from dataset.proposizionalizer import proposizionalize


def make_train_set(clients_df, products_df, orders_df, from_ts, to_ts):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, long, long) -> (dict, np.ndarray, np.ndarray)
    mask = (orders_df['datetime'] >= from_ts) & (orders_df['datetime'] <= to_ts)
    train_df = orders_df.loc[mask]

    # Dizionario di matrici con gli ordini
    orders_dict = make_order_matrices(client_count=clients_df.shape[0],
                                      product_count=products_df.shape[0],
                                      order_df=train_df,
                                      from_ts=from_ts,
                                      to_ts=to_ts)

    prop_train_df = proposizionalize(orders=orders_dict, clients=clients_df, products=products_df)

    # Dati di train sotto forma di ndarray, senza la colonna ordered
    X = prop_train_df.drop('ordered', axis=1).as_matrix()
    # Etichette per i dati di train otto forma di ndarray (colonna ordered)
    y = prop_train_df['ordered'].as_matrix()
    # Dati di train in formato matriciale
    X_dict = orders_dict

    return X_dict, X, y


def make_test_set(clients_df, products_df, orders_df, query_ts):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, long) -> (dict, np.ndarray, np.ndarray)
    return make_train_set(clients_df, products_df, orders_df, query_ts, query_ts)

    #matrix_shape = (clients_df.shape[0], products_df.shape[0])
    #orders_dict = {query_ts: np.zeros(shape=matrix_shape)}
    #return orders_dict, X, y