# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math

from datetime import datetime, date
from sklearn.linear_model import SGDRegressor

from predictor.proposizionalizer import proposizionalize

SECS_IN_DAY = 60 * 60 * 24

# TODO SISTEMARE QUESTO SCHIFO DI CODICE

class MultiRegressorPredictor(object):
    """
    Predittore che utilizza più regressori per stimare il variare delle distribuzioni in probabilità in base al tempo
    trascorso.
    - Un regressore per ogni clienti che stima come varia la probabilità che il cliente faccia un ordine
    - Un regressore per ogni prodotto che stima come varia la probabilità che un prodotto sia ordinato
    - Un regressore che stima come varia la probabilità generale che ci sia un'ordine
    Totale: N_clienti + N_prodotti + 1 regressori
    La probabilità viene poi scalata per la likelyness, ovvero il numero medio di volte che il cliente ha ordinato
    il prodotto.
    """

    def __init__(self, components):
        # type: ([str]) -> None
        self.matrices = {}  # type: dict - dizionario contenente le matrici degli ordini
        self.avg_ones = None  # type: float
        self.clients = None
        self.products = None
        self.pcp_estimation = None
        self.client_regressors = None
        self.product_regressors = None
        self.period_regressor = None
        self.components = components

    def fit(self, clients, products, matrices):
        # type: (pd.DataFrame, pd.DataFrame, dict) -> None
        """
        Inizializza il predittore
        :param matrices: dizionario di matrici degli ordini
        :return:
        """
        self.matrices = matrices
        self.products = products
        self.clients = clients

        sample = matrices[matrices.keys()[0]]
        self.pcp_estimation = np.zeros(shape=sample.shape)

        clients_count = sample.shape[0]
        products_count = sample.shape[1]

        self.product_regressors = np.ndarray(shape=(products_count,), dtype=object)
        self.client_regressors = np.ndarray(shape=(clients_count,), dtype=object)

        days_count = len(matrices.keys())

        # Calcola il numero giornaliero di ordini medio
        ones_cnt = 0
        for day in matrices.keys():
            matrix = matrices[day]
            ones_cnt += matrix.sum()
            for c in range(0, clients_count):
                for p in range(0, products_count):
                    self.pcp_estimation[c, p] += matrices[day][c, p] / days_count
        avg = float(ones_cnt) / float(len(matrices.keys()))
        avg = int(math.ceil(avg))
        self.avg_ones = 1 if avg == 0 else avg

        orders = proposizionalize(matrices, clients, products)

        for index, df in orders.groupby(['client_id']):
            # print index, df.head()
            sums = df.groupby('datetime').sum().reset_index()
            sums['clientId'] = index
            sums = sums[['clientId', 'datetime', 'day_of_year', 'year', 'ordered']]
            # print sums

            train_df = sums.join(clients, on='clientId', rsuffix='_c')
            # print train_df
            y = train_df['ordered'].as_matrix()
            X = train_df.drop(['ordered', 'client_name'], axis=1).as_matrix()
            clf = SGDRegressor()
            clf.fit(X, y)
            self.client_regressors[index] = clf

        for index, df in orders.groupby(['product_id']):
            # print index, df.head()
            sums = df.groupby('datetime').sum().reset_index()
            sums['productId'] = index
            sums = sums[['productId', 'datetime', 'day_of_year', 'year', 'ordered']]
            # print sums

            train_df = sums.join(products, on='productId', rsuffix='_c')
            # print train_df
            y = train_df['ordered'].as_matrix()
            X = train_df.drop(['ordered', 'product_name'], axis=1).as_matrix()
            clf = SGDRegressor()
            clf.fit(X, y)
            self.product_regressors[index] = clf

        sums = orders.groupby('datetime').sum().reset_index()
        # sums['productId'] = index
        sums = sums[['datetime', 'ordered']]
        #print sums
        sums['order_date'] = sums.apply(lambda row: datetime.fromtimestamp(row['datetime']), axis=1)
        sums['year'] = sums.apply(lambda row: row['order_date'].year, axis=1)

        sums['day_of_year'] = sums.apply(lambda row: row['order_date'].toordinal()
                                                     - date(row['order_date'].year, 1, 1).toordinal() + 1, axis=1)
        sums = sums.drop('order_date', axis=1)
        #print sums

        y = sums['ordered'].as_matrix()
        X = sums.drop(['ordered'], axis=1).as_matrix()

        self.period_regressor = SGDRegressor()
        self.period_regressor.fit(X, y)

        return

    def __prepare_dataset_for_prediction(self, clients, products, orders):
        # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray)
        X_client = None
        X_product = None
        for index, df in orders.groupby(['client_id']):
            # print index, df.head()
            sums = df.groupby('datetime').sum().reset_index()
            sums['clientId'] = index
            sums = sums[['clientId', 'datetime', 'day_of_year', 'year', 'ordered']]
            # print sums

            train_df = sums.join(clients, on='clientId', rsuffix='_c')
            # print train_df
            X_client = train_df.drop(['ordered', 'client_name'], axis=1).as_matrix()

        for index, df in orders.groupby(['product_id']):
            # print index, df.head()
            sums = df.groupby('datetime').sum().reset_index()
            sums['productId'] = index
            sums = sums[['productId', 'datetime', 'day_of_year', 'year', 'ordered']]
            # print sums

            train_df = sums.join(products, on='productId', rsuffix='_c')
            # print train_df
            X_product = train_df.drop(['ordered', 'product_name'], axis=1).as_matrix()


        sums = orders.groupby('datetime').sum().reset_index()
        # sums['productId'] = index
        sums = sums[['datetime', 'ordered']]
        #print sums
        sums['order_date'] = sums.apply(lambda row: datetime.fromtimestamp(row['datetime']), axis=1)
        sums['year'] = sums.apply(lambda row: row['order_date'].year, axis=1)

        sums['day_of_year'] = sums.apply(lambda row: row['order_date'].toordinal()
                                                     - date(row['order_date'].year, 1, 1).toordinal() + 1, axis=1)
        sums = sums.drop('order_date', axis=1)
        X_order = sums.drop(['ordered'], axis=1).as_matrix()

        return X_client, X_product, X_order

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
        X_client, X_product, X_period = self.__prepare_dataset_for_prediction(client_data, product_data, query)

        p = 1
        p_c = self.client_regressors[c].predict(X_client)
        p_p = self.product_regressors[p].predict(X_product)
        p_t = self.period_regressor.predict(X_period)
        p_cp = self.pcp_estimation[c,p]
        if 'p_c' in self.components:
            p *= p_c
        if 'p_p' in self.components:
            p *= p_p
        if 'p_t' in self.components:
            p *= p_t
        if 'p_cp' in self.components:
            p *= p_cp
        return p

    def predict_with_threshold(self, order_timestamp, threshold):
        # type: (long, float) -> np.ndarray or None
        """
        Predice un ordine (1) se il corrispondente peso della matrice weights è maggiore del threshold passato
        come parametro
        :param order_timestamp: (long) timestamp della data dell'ordine
        :param threshold: (float) soglia sopra la quale prevedere un 1
        :return: (np.ndarray) matrice degli ordini relativa al timestamp
        """
        if self.pcp_estimation is None:
            return None

        predictions = np.zeros(shape=self.pcp_estimation.shape)
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
        if self.pcp_estimation is None:
            return None

        predictions = np.zeros(shape=self.pcp_estimation.shape)
        clients_count = predictions.shape[0]
        products_count = predictions.shape[1]

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
        if self.pcp_estimation is None:
            return None

        predictions = np.zeros(shape=self.pcp_estimation.shape)
        clients_count = predictions.shape[0]
        products_count = predictions.shape[1]

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c, p] = self.__calculate_order_probability(c, p, order_timestamp)

        vectorized_weights = np.reshape(probs, (1, probs.size))  # (1x Nweights)
        result = np.ones(shape=(probs.size, 2))
        result[:, 0] = result[:, 0] - vectorized_weights
        result[:, 1] = vectorized_weights
        return result
