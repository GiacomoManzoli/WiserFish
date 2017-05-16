# -*- coding: utf-8 -*-
import csv
import json

import pandas as pd
import numpy as np
import pickle
import os

# Nomi delle directory
D_OUTPUTS = "data"
D_TRAIN = "train_set"
D_TEST = "test_set"
D_VERSION = "version_"


def __output_dir(prefix, make_dirs=True):
    # type: (str, bool) -> (str, str, str)
    """
    Crea le cartelle in cui andare a salvare i vari file
    :param prefix: 
    :param make_dirs: 
    :return: tupla contenente i filepath creati
    """
    experiment_path = "./%s/%s/" % (D_OUTPUTS, prefix)
    experiment_path_train = "%s%s/" % (experiment_path, D_TRAIN)
    experiment_path_test = "%s%s/" % (experiment_path, D_TEST)

    if not os.path.exists(experiment_path) and make_dirs:
        os.makedirs(experiment_path)
    if not os.path.exists(experiment_path_train) and make_dirs:
        os.makedirs(experiment_path_train)
    if not os.path.exists(experiment_path_test) and make_dirs:
        os.makedirs(experiment_path_test)
    return experiment_path, experiment_path_train, experiment_path_test


################################################################
# AUX FUNCTIONS
################################################################

def __save_orders(orders, clients, products, filename):
    # type: (dict, pd.DataFrame, pd.DataFrame, str) -> None
    """
    Salva gli ordini su file. Vengono salvati solo gli ordini effettuati (gli 1 della matrice) con
    il timestamp e gli id dei relativi clienti/prodotti
    :param orders: dizionario di matrici con gli ordini
    :param clients: dataframe con i clienti
    :param products: dataframe con i prodotti
    :param filename: nome del file
    """
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
    df_cols = ['datetime', 'clientId', 'productId']
    if len(orders_rows) != 0:
        orders_df = pd.DataFrame(orders_rows, columns=df_cols)
        orders_df['datetime'] = orders_df['datetime'].astype(dtype=long)
        orders_df['clientId'] = orders_df['clientId'].astype(dtype=int)
        orders_df['productId'] = orders_df['productId'].astype(dtype=int)
        orders_df.to_csv(filename, index=False)
    else:
        # writes an empty file
        with open(filename, 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(df_cols)


def __load_orders(filename, matrix_shape):
    # type: (str, (int, int)) -> dict
    """
    Carica gli ordini dal file e li ritorna nel formato `dizionario di matrici`
    :param filename: percorso del file da caricare
    :param matrix_shape: coppia rappresentante le dimensioni della matrice
    :return: ordini in formato dizionario di matrici
    """
    try:
        orders = pd.read_csv(filename)
        orders_dict = {}
        for idx, row in orders.iterrows():
            key = int(row['datetime'])  # timestamp dell'ordine
            if key not in orders_dict.keys():
                orders_dict[key] = np.zeros(shape=matrix_shape)

            orders_dict[key][int(row['clientId']), int(row['productId'])] = 1
    except IOError:
        print "Couldn't load the orders"
        orders_dict = {}
    return orders_dict


################################################################
# SAVE AND LOAD OF THE TRAIN DATASET
################################################################

def save_train_set(clients, products, orders, model, name='', prefix=''):
    # type: (pd.DataFrame, pd.DataFrame, dict, str, str) -> None
    """
    Salva i dataframe dei clienti, dei prodotti e le matrici degli ordini in vari file CSV dentro
    la directory `data`.
    VERSIONE TRAIN
    :param clients: dataframe dei clienti
    :param products: dataframe dei prodotti
    :param orders: dizionario di matrici con gli ordini
    :param model: modello di probabilità utilizzato per generare il dataset
    :param name: nome del dataset
    :param prefix: prefisso da utilizzare
    """
    if name == '':
        name = prefix
    base_path, train_set_path, _ = __output_dir(name)

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    print "Saving common data into:", base_path
    clients.to_csv('%s%sclients.csv' % (base_path, prefix), index=False)
    products.to_csv('%s%sproducts.csv' % (base_path, prefix), index=False)
    with open('%s%spmodel.pkl' % (base_path, prefix), 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    print "Saving TRAIN SET into:", train_set_path, prefix
    orders_file_name = '%s%sorders.csv' % (train_set_path, prefix)
    __save_orders(orders, clients, products, orders_file_name)


def load_train_set(name='', prefix=''):
    # type: (str, str) -> (pd.DataFrame, pd.DataFrame, dict, ProbabilityModel)
    """
    Carica il dataset per il training
    :param name: nome del dataset
    :param prefix: prefisso
    :return: il dataset sotto forma di dataframe e di dizionario di matrici
    """
    if name == '':
        name = prefix
    base_path, train_set_path, _ = __output_dir(name)

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    print "Loading from:", base_path, prefix
    clients = pd.read_csv('%s%sclients.csv' % (base_path, prefix))
    products = pd.read_csv('%s%sproducts.csv' % (base_path, prefix))
    with open('%s%spmodel.pkl' % (base_path, prefix), 'rb') as input_file:
        model = pickle.load(input_file)

    orders_shape = (clients.shape[0], products.shape[0])
    orders_filename = '%s%sorders.csv' % (train_set_path, prefix)
    orders_dict = __load_orders(orders_filename, orders_shape)

    return clients, products, orders_dict, model


################################################################
# SAVE AND LOAD OF THE TEST DATASET
################################################################

def save_test_set(clients, products, orders, name='', prefix='', version=None):
    # type: (pd.DataFrame, pd.DataFrame, dict, str, str, int) -> None
    """
    Salva i dataframe dei clienti, dei prodotti e le matrici degli ordini in vari file CSV dentro
    la directory `data`.
    VERSIONE TEST: non vengono salvati i clienti/prodotti/modello
    :param clients: dataframe dei clienti
    :param products: dataframe dei prodotti
    :param orders: dizionario di matrici con gli ordini
    :param model: modello di probabilità utilizzato per generare il dataset
    :param name: nome del dataset
    :param prefix: prefisso dei file
    :param version: versione del dataset (possono essere state generate più matrici per lo stesso giorno)
    """
    if name == '':
        name = prefix
    _, _, test_set_path = __output_dir(name)

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    if version is not None:
        test_set_path = "%s%s%d/" % (test_set_path, D_VERSION, version)
        if not os.path.exists(test_set_path):
            os.makedirs(test_set_path)

    print "Saving TEST SET into:", test_set_path, prefix
    orders_file_name = '%s%sorders.csv' % (test_set_path, prefix)
    __save_orders(orders, clients, products, orders_file_name)


def load_test_set(name='', prefix='', version=None):
    # type: (str, str, int) -> (pd.DataFrame, pd.DataFrame, dict, ProbabilityModel)
    """
    Carica il dataset per il test
    :param name: nome del dataset
    :param prefix: prefisso
    :param version: versione del dataset da caricare
    :return: il dataset sotto forma di dataframe e di dizionario di matrici
    """
    if name == '':
        name = prefix
    base_path, _, test_set_path = __output_dir(name)

    if version is not None:
        test_set_path = "%sversion_%d/" % (test_set_path, version)

    if prefix != '' and prefix[-1] != '_':
        prefix += '_'

    print "Loading from:", base_path, prefix
    clients = pd.read_csv('%s%sclients.csv' % (base_path, prefix))
    products = pd.read_csv('%s%sproducts.csv' % (base_path, prefix))
    with open('%s%spmodel.pkl' % (base_path, prefix), 'rb') as input_file:
        model = pickle.load(input_file)

    orders_shape = (clients.shape[0], products.shape[0])
    orders_filename = '%s%sorders.csv' % (test_set_path, prefix)
    orders_dict = __load_orders(orders_filename, orders_shape)

    return clients, products, orders_dict, model


################################################################
# PARTIALS LOADING/SAVING
################################################################

def merge_partial_orders(name, prefixes, result_prefix):
    # type: (str, [str], str) -> None
    """
    Carica da vari CSV contenenti le parti di un dataset e crea la versione completa.
    Il risultato viene salvato su file.
    :param name:
    :param prefixes:
    :param result_prefix:
    :return:
    """
    assert len(prefixes) >= 1
    base_path, _, _ = __output_dir(name, False)

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


################################################################
# CONFIGURATION FILE HELPER
################################################################

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
