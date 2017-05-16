# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import linear_model
import numpy as np
import math

from sklearn.svm import SVR

from predictor.proposizionalizer import proposizionalize

SECS_IN_DAY = 60*60*24
# TODO SISTEMARE QUESTO SCHIFO DI CODICE

class SingleRegressorPredictor(object):
    """
    Predittore che usa un regressore per ogni cella della matrice clienti-prodotti per approssimare
    la probabilità di una vendita al variare del periodo.
    I regressori possono essere addestrati sui dati giornalieri oppure aggregando i giorni.
    """

    def __init__(self, group_size=5, regressor_name='SGD'):
        # type: (int, str) -> None
        self.regressor_name = regressor_name
        self.avg_ones = None
        self.group_size = group_size
        self.regressor_matrix = None
        self.clients = None
        self.products = None

    def __extract_day_group(self, dataset):
        # type: (pd.DataFrame) -> pd.DataFrame
        """
        Aggiunge la colonna `day_group` al dataset.
        Vengono rimossse le colonne: `datetime`, `year` e `day_of_year`.
        :param dataset: dataset al quale aggiungere la colonna `day_group`
        :return: 
        """
        df = dataset
        df['day_group'] = df['datetime'] / (SECS_IN_DAY * self.group_size)
        df['day_group'] = df['day_group'].astype(int)
        df = df.drop(['datetime', 'year', 'day_of_year'], axis=1)
        return df

    def fit(self, clients, products, matrices):
        # type: (pd.DataFrame, pd.DataFrame, dict) -> None
        """
        Inizializza il predittore
        :param clients: dataframe con i dati dei clienti
        :param products: dataframe con i dati dei prodotti
        :param matrices: dizionario con le matrici degli ordini
        :return: 
        """
        self.clients = clients
        self.products = products
        dataset = proposizionalize(matrices, clients, products)

        client_cnt = dataset['client_id'].nunique()
        product_cnt = dataset['product_id'].nunique()

        self.regressor_matrix = np.ndarray(shape=(client_cnt, product_cnt), dtype=object)
        df = self.__extract_day_group(dataset)

        for index, group in df.groupby(['client_id', 'product_id']):
            c, p = index
            # print "Fitting the Regressor for client", c, "product", p
            group = group.groupby('day_group').mean()  #
            # df['ordered'] è nel range [0,1] ed è la probabilità stimata che ci sia almeno
            # un ordine nel day_group
            # Fissato un day_group tutti i valori del dataframe, eccetto la colonna `ordered` sono
            # costanti, quindi farne la media non altera il valore.
            group = group.reset_index()

            X = group.drop(['ordered'], axis=1).as_matrix()
            y = group['ordered'].as_matrix()

            clf = None
            if self.regressor_name == 'SGD':
                clf = linear_model.SGDRegressor()
            elif self.regressor_name == 'SVR':
                clf = SVR()

            self.regressor_matrix[c, p] = clf
            self.regressor_matrix[c, p].fit(X, y)

        ones_cnt = 0  # counts the total number of orders
        for day in matrices.keys():
            ones_cnt += matrices[day].sum()

        avg = float(ones_cnt) / float(len(matrices.keys()))
        avg = int(math.ceil(avg))
        self.avg_ones = 1 if avg == 0 else avg
        return

    def __calculate_order_probability(self, c, p, timestamp):
        # type: (int, int, long) -> float
        """
        Calcola la probabilità che il cliente c ordini il prodotto p nel periodo t
        :param c: (int) posizione del cliente nella matrice (coincide con l'id)
        :param p: (int) posizione del prodotto nella matrice (coincide con l'id) 
        :param t: (long) timestamp dell'ordine
        :return: probabilità che venga effettuato l'ordine
        """
        client_data = pd.DataFrame([self.clients.iloc[c]])
        product_data = pd.DataFrame([self.products.iloc[p]])

        fake_orders = {
            timestamp: np.zeros(shape=(1, 1))
        }
        query = proposizionalize(fake_orders, client_data, product_data)
        query = query.drop('ordered', axis=1)
        query = self.__extract_day_group(query)
        # print query
        return self.regressor_matrix[c, p].predict(query.as_matrix())

    def predict_with_threshold(self, order_timestamp, threshold):
        # type: (long, float) -> np.ndarray or None
        """
        Predice un ordine (1) se il corrispondente peso della matrice weights è maggiore del threshold passato
        come parametro
        :param order_timestamp: (long) timestamp della data dell'ordine
        :param threshold: (float) soglia sopra la quale prevedere un 1
        :return: (np.ndarray) matrice degli ordini relativa al timestamp
        """
        if self.regressor_matrix is None:
            return None

        predictions = np.zeros(shape=self.regressor_matrix.shape)
        clients_count = predictions.shape[0]
        products_count = predictions.shape[1]

        # check threshold
        for c in range(0, clients_count):
            for p in range(0, products_count):
                order_probability = self.__calculate_order_probability(c, p, order_timestamp)
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
        if self.regressor_matrix is None:
            return None
        clients_count = self.regressor_matrix.shape[0]
        products_count = self.regressor_matrix.shape[1]

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c, p] = self.__calculate_order_probability(c, p, order_timestamp)

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
        if self.regressor_matrix is None:
            return None
        clients_count = self.regressor_matrix.shape[0]
        products_count = self.regressor_matrix.shape[1]

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c, p] = self.__calculate_order_probability(c, p, order_timestamp)

        vectorized_weights = np.reshape(probs, (1, probs.size))  # (1x Nweights)
        result = np.ones(shape=(probs.size, 2))
        result[:, 0] = result[:, 0] - vectorized_weights
        result[:, 1] = vectorized_weights
        return result
