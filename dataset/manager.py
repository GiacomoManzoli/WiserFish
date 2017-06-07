# -*- coding: utf-8 -*-
import fnmatch
import os, csv, pickle
from shutil import copyfile

import pandas as pd
import numpy as np

from dataset.generator.client import Client
from dataset.generator.order import Order
from dataset.generator.polynomial import PeriodicPolynomial
from dataset.generator.product import Product
from dataset.generator.values import SECS_IN_DAY
from util import ConfigurationFile

D_OUTPUTS = "datasets"  # directory dove mettere i risultati
D_GENERATORS = "generators"  # sotto-directory dove mettere i vari pickle dei generatori
D_GENERATORS_PROD = "products"
D_GENERATORS_CLIENT = "clients"
D_DATA = "data"  # sotto-directory dove mettere i csv generati
D_DATA_PARTIALS = "parts"


def save_configuration_file(config):
    # type: (ConfigurationFile) -> None
    base_path, _, _ = __output_dirs(config.base_prefix)
    copyfile(config.file_path, base_path+config.base_prefix+".json")


def save_dataset(clients, products, orders, global_trend, dataset_name='sample'):
    # type: ([Client], [Product], [Order], PeriodicPolynomial) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    Salva il dataset generato e i corrispettivi generatori nella directory `dataset`.
    Ritorna i DataFrame che sono stati salvati.
    Da notare che nei DataFrame salvati NON vengono messi i dati utili a generare gli ordini,
    ma solo quelli dai utilizzare in fase di apprendimento.
    I dati con utili per la generazione vengono salvati su dei file a parte utilizzando pickle.
    :param clients: Clienti che hanno generato il dataset 
    :param products: Prodotti presenti nel dataset
    :param orders: Ordini generati dai clienti
    :param dataset_name: Nome del dataset, corrisponde con il nome della directory dove salvare i dati
    """
    base_path, generators_path, data_path = __output_dirs(dataset_name)

    # Salva il trend globale
    with open(generators_path+"global_trend.pkl", 'wb') as output:
        pickle.dump(global_trend, output, pickle.HIGHEST_PROTOCOL)

    # Salva i prodotti (sia come pickle che come dataframe)
    product_rows = []
    for p in products:
        file_name = generators_path + D_GENERATORS_PROD + "/" + p.name + ".pkl"
        with open(file_name, 'wb') as output:
            pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)
        # riga per il dataframe
        product_rows += [p.to_dict()]
    product_df = pd.DataFrame(product_rows, columns=Product.get_dataframe_header())
    product_df.to_csv(data_path + 'products.csv', index=False)

    # Salva i clienti (sia come pickle che come dataframe)
    client_rows = []
    for c in clients:
        file_name = generators_path + D_GENERATORS_CLIENT + "/" + c.name + ".pkl"
        with open(file_name, 'wb') as output:
            pickle.dump(c, output, pickle.HIGHEST_PROTOCOL)
        # riga per il dataframe
        client_rows += [c.to_dict()]
    client_df = pd.DataFrame(client_rows, columns=Client.get_dataframe_header())
    client_df.to_csv(data_path + 'clients.csv', index=False)

    # Salva gli ordini (solo come dataframe)
    orders_filename = data_path + 'orders.csv'
    __save_orders(orders, orders_filename)
    return


def save_partial_orders(dataset_name, orders, part_index):
    _, _, data_path = __output_dirs(dataset_name)
    filename = data_path + D_DATA_PARTIALS + "/orders_part_%d.csv" % part_index
    __save_orders(orders, filename)


def merge_partial_orders_and_save(dataset_name, parts_count):
    _, _, data_path = __output_dirs(dataset_name)

    order_df = None

    for part in range(0, parts_count):
        filename = data_path + D_DATA_PARTIALS + "/orders_part_%d.csv" % part
        if order_df is None:
            order_df = pd.read_csv(filename)
        else:
            order_df = pd.concat([order_df, pd.read_csv(filename)])
    order_df['datetime'] = order_df['datetime'].astype(dtype=long)
    order_df['client_id'] = order_df['client_id'].astype(dtype=int)
    order_df['client_id'] = order_df['client_id'].astype(dtype=int)
    order_df = order_df.sort_values('datetime')
    order_df.to_csv(data_path + 'orders.csv', index=False)


def save_generators(clients, products, global_trend, dataset_name='sample'):
    """
    Salva solo gli oggetti necessari alla generazione del dataset
    :param clients: 
    :param products: 
    :param global_trend: 
    :param dataset_name: 
    :return: 
    """
    save_dataset(clients, products, [], global_trend, dataset_name)


def load_generators(dataset_name='sample'):
    _, generators_path, _ = __output_dirs(dataset_name, make_dirs=False)
    clients = []
    for file in os.listdir(generators_path+D_GENERATORS_CLIENT):
        if fnmatch.fnmatch(file, 'client_*.pkl'):
            print(generators_path+D_GENERATORS_CLIENT+"/"+file)
            with open(generators_path+D_GENERATORS_CLIENT+"/"+file, 'rb') as input:
                clients += [pickle.load(input)]

    clients.sort(key=lambda c: c.id)
    assert clients[0].id == 0

    products = []
    for file in os.listdir(generators_path+D_GENERATORS_PROD):
        if fnmatch.fnmatch(file, 'product_*.pkl'):
            print(generators_path+D_GENERATORS_PROD+"/"+file)
            with open(generators_path+D_GENERATORS_PROD+"/"+file, 'rb') as input:
                products += [pickle.load(input)]

    global_trend = None
    with open(generators_path+'global_trend.pkl', 'rb') as input:
        global_trend = pickle.load(input)

    return clients, products, global_trend


def load_dataset(dataset_name='sample'):
    # type: (str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    Carica i DataFrame relativi al dataset passato per parametro
    :param dataset_name: nome del dataset
    :return: DataFrame dei clienti, prodotti e ordini
    """
    _, _, data_path = __output_dirs(dataset_name, make_dirs=False)

    client_df = pd.read_csv(data_path + 'clients.csv')
    product_df = pd.read_csv(data_path + 'products.csv')
    order_df = pd.read_csv(data_path + 'orders.csv')

    return client_df, product_df, order_df


def make_order_matrices(client_count, product_count, order_df, from_ts, to_ts):
    # type: (int, int, pd.DataFrame) -> dict
    """
    Crea il dizionario di matrici degli ordini a partire da un dataframe degli ordini
    :param client_count: numero di clienti presenti nel dataset
    :param product_count: numero di prodotti presenti nel dataset
    :param order_df: dataframe con gli ordini
    :return: dizionario indicizzato per timestamp contenente le varie matrici
    """
    order_dict = {}
    matrix_shape = (client_count, product_count)

    for ts in range(from_ts, to_ts+SECS_IN_DAY, SECS_IN_DAY):
        order_dict[ts] = np.zeros(shape=matrix_shape)

    for idx, row in order_df.iterrows():
        key = int(row['datetime'])  # timestamp dell'ordine
        assert key in order_dict.keys()
        #if key not in order_dict.keys():
        #    order_dict[key] = np.zeros(shape=matrix_shape)
        order_dict[key][int(row['client_id']), int(row['product_id'])] = 1

    return order_dict


# Metodi ausiliari

def __output_dirs(prefix, make_dirs=True):
    # type: (str, bool) -> (str, str, str)
    """
    Crea le cartelle in cui andare a salvare i vari file
    :param prefix: 
    :param make_dirs: 
    :return: tupla contenente i filepath creati
    """
    experiment_path = "./%s/%s/" % (D_OUTPUTS, prefix)
    experiment_path_generators = "%s%s/" % (experiment_path, D_GENERATORS)
    experiment_path_generators_prod = \
        "%s%s/" % (experiment_path_generators, D_GENERATORS_PROD)
    experiment_path_generators_client = \
        "%s%s/" % (experiment_path_generators, D_GENERATORS_CLIENT)
    experiment_path_data = "%s%s/" % (experiment_path, D_DATA)
    experiment_path_data_parts = "%s%s/" % (experiment_path_data, D_DATA_PARTIALS)
    if not os.path.exists(experiment_path) and make_dirs:
        os.makedirs(experiment_path)

    if not os.path.exists(experiment_path_generators) and make_dirs:
        os.makedirs(experiment_path_generators)
        os.makedirs(experiment_path_generators_prod)
        os.makedirs(experiment_path_generators_client)

    if not os.path.exists(experiment_path_data) and make_dirs:
        os.makedirs(experiment_path_data)
        os.makedirs(experiment_path_data_parts)
    return experiment_path, experiment_path_generators, experiment_path_data


def __save_orders(orders, orders_filename):
    # type: ([Order], str) -> None
    order_rows = []
    for o in orders:
        order_rows.append(o.to_dict())

    if len(order_rows) != 0:
        order_df = pd.DataFrame(order_rows, columns=Order.get_dataframe_header())
        order_df['datetime'] = order_df['datetime'].astype(dtype=long)
        order_df['client_id'] = order_df['client_id'].astype(dtype=int)
        order_df['client_id'] = order_df['client_id'].astype(dtype=int)
        order_df = order_df.sort_values('datetime')
        order_df.to_csv(orders_filename, index=False)
    else:
        # writes an empty file
        with open(orders_filename, 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(Order.get_dataframe_header())
