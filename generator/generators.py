# -*- coding: utf-8 -*-
import datetime
import random
import time

from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

from generator.probability_models import RandomProbabilityModel, SimpleProbabilityModel, CondProbabilityModel, \
    ProbabilityModel, NoisyProbabilityModel

SECS_IN_DAY = 60*60*24


def generate_probabilites(y):
    # type: (np.array) -> np.array
    """
    Crea un array di valori di probabilità associato ad ognuno dei valori presenti nell'array `y`.
    Il range nel quale viene generato ogni singolo valori dipende dal corrispettivo valore di `y`. 
    """
    class_cnt = np.bincount(y).size
    prob_interval = 1 / float(class_cnt)
    probs = np.array(y, dtype=np.float)

    for i in range(0, y.size):
        probs[i] = random.uniform(prob_interval * y[i], prob_interval * (y[i]+1))

    return probs


def generate_freq_scales(y):
    # type: (np.array) -> np.array
    """
    Genere un arrau di scale di freuqenza, una per ogni istanza di `y`.
    Il valore generato è influenzato dalla classe dell'istanza.
    """
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
    """
    Genera i clienti
    :param cnt: numero di clienti da generare
    :return: dataframe contenente i clienti generati
    """
    features = 10
    (X, y) = make_classification(n_samples=cnt, n_features=features, n_informative=4, n_repeated=2, n_classes=4,
                                 random_state=123)
    # X[i] array rappresentante l'i-esimo cliento, y[i] è la classe del cliente (4 possibili valori)

    # 4 diverse classi per la probabilità di effettuare un ordine per il cliente (determina pc)
    clients_order_probs = [random.randint(0, 3) for _ in range(0, cnt)]
    # 4 diverse classi per la frequenza d'ordine (determina `client_freq`)
    clients_order_freqs = [random.randint(0, 3) for _ in range(0, cnt)]

    # vettore degli id
    ids = [i for i in range(0, cnt)]
    # vettore con i nomi
    names = ["client_" + str(i) for i in range(0, cnt)]

    # probabilità di effettuare un ordine
    pc = generate_probabilites(np.array(clients_order_probs))
    # frequenza degli ordini
    freq_scale = generate_freq_scales(np.array(clients_order_freqs))

    # creazione del dataframe
    merged = np.c_[ids, pc, freq_scale, X, y]
    clients = pd.DataFrame(names, columns=["client_name"])  # non posso aggiungere subito i nomi

    col_names = ["clientId", "pc", "client_freq_scale"] + ["client_feature_" + str(i) for i in range(0, features)] + ["class"]
    clients_data = pd.DataFrame(merged, columns=col_names)

    clients = clients.join(clients_data)

    clients['clientId'] = clients['clientId'].astype(dtype=int)
    clients['class'] = clients['class'].astype(dtype=int)
    return clients


###########################
# PRODUCTS DATA
###########################

def generate_products(cnt):
    # type: (int) -> pd.DataFrame
    """
    Genera i prodotti
    :param cnt: numero di prodotti da generare
    :return: dataframe contenente i prodotti 
    """
    features = 15
    (X, y) = make_classification(n_samples=cnt, n_features=features, n_informative=4, n_repeated=2, n_classes=4,
                                 random_state=456)
    # X[i] array rappresentate l'i-esimo prodotto, y[i] è la classe del prodotto (4 possibili)

    # 4 diverse classi per la probabilità del prodotto di essere ordinato (determina pp)
    products_order_probs = [random.randint(0, 3) for _ in range(0, cnt)]
    # 4 diverse classi per la frequenza d'ordine (determina `product_freq`)
    products_order_freqs = [random.randint(0, 3) for _ in range(0, cnt)]

    ids = [i for i in range(0, cnt)]
    names = ["prod_" + str(i) for i in range(0, cnt)]

    # vettore con le probabilità di essere ordinato
    pp = generate_probabilites(np.array(products_order_probs))
    # vettore con le frequenze
    freq_scale = generate_freq_scales(np.array(products_order_freqs))

    # crea il dataframe
    merged = np.c_[ids, pp, freq_scale, X, y]
    products = pd.DataFrame(names, columns=["product_name"])

    col_names = ["productId", "pp", "product_freq_scale"] + ["product_feature_" + str(i) for i in range(0, features)] + ["class"]
    products_data = pd.DataFrame(merged, columns=col_names)

    products = products.join(products_data)

    # Imposta i tipi corretti
    products['productId'] = products['productId'].astype(dtype=int)
    products['class'] = products['class'].astype(dtype=int)
    return products


###########################
# ORDER DATA
###########################

def generate_matrix(clients, products, timestamp, model):
    # type: (pd.DataFrame, pd.DataFrame, long, ProbabilityModel) -> np.ndarray
    """
    Genera la matrice degli ordini relativa al timestamp
    :param clients: dataframe con i clienti
    :param products: dataframe con i prodotti
    :param timestamp: timestamp della data a cui sono associati gli ordini
    :param model: modello di probabilità da utilizzare
    :return: matrice degli ordini
    """
    clients_count = clients.shape[0]
    products_count = products.shape[0]
    matrix = np.ndarray(shape=(clients_count, products_count))

    for c in range(0, clients_count):
        for p in range(0, products_count):
            matrix[c, p] = model.will_make_order(clients.ix[c], products.ix[p], timestamp)
    return matrix


def generate_orders(clients, products, days, model):
    # type: (pd.DataFrame, pd.DataFrame, [long], ProbabilityModel) -> dict
    """
    Genera il dizionario di matrici degli ordini per le matrici specificate come parametro
    :param clients: dataframe con i clienti
    :param products: dataframe con i prodotti
    :param days: timestamp dei giorni
    :param model: modello di probabilità da utilizzare
    :return: dizionario con le matrici degli ordini
    """
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
    # type: (int, int) -> [long]
    """
    Genera i timestamp per i giorni a partire da ieri (rispetto la data corrente) a ritroso.
    :param days_count: numero di timestamp da generare
    :param day_interval: giorni di stacco tra i timestamp generati (0 -> contigui)
    :return: lista con i timestamp generati
    """
    return [time.time() - SECS_IN_DAY * i * (day_interval + 1) for i in range(1, days_count+1)]


def generate_dataset(clients_count, products_count, days_count, day_interval=0, model_name='random'):
    # type: (int, int, int, int, str) -> (pd.DataFrame, pd.DataFrame, dict, ProbabilityModel)
    """
    Genera un dataset per il problema
    :param clients_count: numero di clienti
    :param products_count: numero di prodotti
    :param days_count: nummero di giorni (matrici)
    :param day_interval: salto tra i giorni (default=0, tutti di fila)
    :param model_name: nome del modello di probabilità {'random','simple', 'cond'}
    :return: dataset
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
    elif model_name == 'noisy':
        model = NoisyProbabilityModel(clients, products)

    if days_count != 0:
        orders = generate_orders(clients, products, days, model)
    else:
        orders = {}

    return clients, products, orders, model


###########################
# UTILITIES
###########################

def __check_generated_days(days):
    # type: ([long]) -> None
    """
    DEBUG: stampa i giorni generati
    :param days: timestamp generati
    """
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