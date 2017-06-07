# -*- coding: utf-8 -*-
import datetime
import pandas as pd
import numpy as np
import math

from dataset.generator.probability_models import CondProbabilityModel


class SinfulBaselinePredictor(object):
    """
    Predittore che usa lo stesso modello di probabilità utilizzato per generare i dati.
    La predizione viene scegliendo gli ordini con una probabilità maggiore (come per gli altri
    classificatori) e NON come nella generazione dei dati (scelta pesata in base alla probabilità).
    """

    def __init__(self):
        self.clients = None
        self.products = None
        self.prob_model = None
        self.avg_ones = None

    def fit(self, clients, products, matrices):
        # type: (pd.DataFrame, pd.DataFrame, dict) -> None
        """
        Inizializza il predittore
        :param products: dataframe con i dati dei clienti
        :param clients: dataframe con i dati dei prodotti
        :param matrices: dizionario di matrici degli ordini
        :return:
        """
        self.clients = clients
        self.products = products
        self.prob_model = CondProbabilityModel(clients, products)

        ones_cnt = 0  # counts the total number of orders
        for day in matrices.keys():
            ones_cnt += matrices[day].sum()

        avg = float(ones_cnt) / float(len(matrices.keys()))
        avg = int(math.ceil(avg))
        self.avg_ones = 1 if avg == 0 else avg
        return

    def predict_with_threshold(self, order_timestamp, threshold):
         # type: (long, float) -> np.ndarray or None
        """
        Predice un ordine (1) se il corrispondente peso della matrice weights è maggiore del threshold passato
        come parametro
        :param order_timestamp: (long) timestamp della data dell'ordine
        :param threshold: (float) soglia sopra la quale prevedere un 1
        :return: (np.ndarray) matrice degli ordini relativa al timestamp
        """
        if self.prob_model is None:
            return None

        clients_count = self.clients.shape[0]
        products_count = self.products.shape[0]

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        t = order_date.timetuple().tm_yday  # Day of the year

        predictions = np.ndarray(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                order_probability = self.prob_model.probability(self.clients.ix[c], self.products.ix[p], t)
                predictions[c, p] = 1 if order_probability >= threshold else 0
        return predictions

    def predict_with_topn(self, order_timestamp):
        # type: (long) -> np.ndarray or None
        """
        Utilizza come threshold per le predizioni un valore tale che vengono predetti tanti ordini quanto è il numero
        medio di orgini che viene effettuato giornalmente
        :param order_timestamp: (long) timestamp della data dell'ordine
        :return: (np.ndarray) matrice degli ordini relativa al timestamp
        """
        if self.prob_model is None:
            return None

        clients_count = self.clients.shape[0]
        products_count = self.products.shape[0]

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c, p] = self.prob_model.probability(self.clients.ix[c], self.products.ix[p], t)

        # copy is needed because reasons... (reshape + sort)
        probs_vec = np.reshape(probs.copy(), probs.size)
        probs_vec[::-1].sort()  # in place revese sort

        threshold = probs_vec[self.avg_ones - 1]
        return self.predict_with_threshold(order_timestamp, threshold)

    def predict_proba(self, order_timestamp):
        # type: (long) -> np.ndarray or None
        """
        Ritorna la matrice contenente le probabilità che vengano effettuati degli ordini nel giorno specificato come
        parametro
        :param order_timestamp: (long) timestamp della data dell'ordine
        :return: (np.ndarray) matrice con le probabilità di effettuare un ordine il giorno specificato dal timestamp.
                 La matrice è nel formato (numero_coppie, 2) dove `numero_coppie` = numero clienti x numero prodotti.
                 La prima colonna della matrice contiene la probabilità della classe 0, mentre la seconda colonna
                 contiene quella della classe 1.
        """
        if self.prob_model is None:
            return None

        clients_count = self.clients.shape[0]
        products_count = self.products.shape[0]

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        t = order_date.timetuple().tm_yday  # Day of the year

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c, p] = self.prob_model.probability(self.clients.ix[c], self.products.ix[p], t)

        vectorized_weights = np.reshape(probs, (1, probs.size))  # (1x Nweights)
        result = np.ones(shape=(probs.size, 2))
        result[:, 0] = result[:, 0] - vectorized_weights
        result[:, 1] = vectorized_weights
        return result
