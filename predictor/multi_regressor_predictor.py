# -*- coding: utf-8 -*-
import math
from datetime import datetime

import numpy as np
import pandas as pd
import sqlite3
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.svm import SVR

from dataset.generator.values import SECS_IN_DAY
from util import Log

TAG = "MultiRegressorPredictor"


class MultiRegressorPredictor(object):
    """
    Predittore che utilizza più regressori per stimare la probabilità che un cliente ordini
    - Un regressore per ogni cliente che stima come varia la probabilità che il cliente faccia un ordine
    - Un regressore per ogni prodotto che stima come varia la probabilità che un prodotto sia ordinato
    - Un regressore che stima come varia la probabilità generale che ci sia un'ordine
    Totale: N_clienti + N_prodotti + 1 regressori
    La probabilità viene poi scalata per la likelyness, ovvero il numero medio di volte che il cliente ha ordinato
    il prodotto.
    """

    CLIENT_TRAIN_COLS = ['client_id', 'datetime', 'day_of_year', 'year']
    PRODUCT_TRAIN_COLS = ['product_id', 'datetime', 'day_of_year', 'year']
    PERIOD_TRAIN_COLS = ['datetime', 'day_of_year', 'year']
    TARGET_COL = 'cnt_ordered'

    def __init__(self, components, regressor_name='SVR'):
        # type: ([str], str) -> None
        self.regressor_name = regressor_name
        self.avg_ones = None  # type: float
        self.pcp_estimation = None
        self.client_regressors = None
        self.product_regressors = None
        self.period_regressor = None
        self.components = components

    def __build_clf(self):
        # type: () -> SVR or SGDRegressor or PassiveAggressiveRegressor or None
        """
        Crea un regressore in base a quanto specificato alla creazione dell'oggetto
        :return: 
        """
        clf = None
        if self.regressor_name == 'SGD':
            clf = linear_model.SGDRegressor()
        elif self.regressor_name == 'SVR':
            clf = SVR()
        elif self.regressor_name == 'PAR':
            clf = PassiveAggressiveRegressor()
        return clf

    @staticmethod
    def __prepare_clients_dataset(cnx, clients_count, from_ts, to_ts):
        # type: (sqlite3.connect, int, long, long) -> dict
        """
        Crea il dataset per il training dei regressori associati ai cklienti, recuperando i dati dalla
        connessione al database e per l'intervallo di tempo specificato dai due timestamp.
        :param cnx: connessione al database da utilizzare
        :param clients_count: numero di clienti presenti
        :param from_ts: timestamp relativo al giorno di partenza del dataset
        :param to_ts: timestamp relativo all'ultimo giorno (incluso) del dataset
        :return: {(X, y)} dizionario indicizzato per id del cliente contenente le coppie (matrice 
         di train, valori associati) da utilizzare per effettuare il train del relativo regressore
        """
        dfs = {}

        for c in range(0, clients_count):
            # Crea il DataFrame con anche le righe per i prodotti non ordinati
            query = "select datetime, count(*) " \
                    "from orders " \
                    "where datetime >= %d and datetime <= %d and client_id = %d " \
                    "group by datetime, product_id " \
                    "order by datetime, client_id, product_id" % (from_ts, to_ts, c)
            Log.d(TAG, query)
            # ^ ORDER BY è fondamentale per effettuare la creazione in modo efficiente
            cursor = cnx.execute(query)

            next_row = cursor.fetchone()
            df_rows = []
            for ts in range(from_ts, to_ts + SECS_IN_DAY, SECS_IN_DAY):
                ordered = 0
                if next_row is not None and next_row[0] == ts:
                    ordered = next_row[1]
                    next_row = cursor.fetchone()

                order_date = datetime.fromtimestamp(ts)
                day_of_year = order_date.timetuple().tm_yday
                year = order_date.timetuple().tm_year
                df_rows.append({
                    'datetime': ts,  # timestamp della data dell'ordine
                    'day_of_year': day_of_year,
                    'year': year,
                    'client_id': c,
                    MultiRegressorPredictor.TARGET_COL: ordered
                })
            df = pd.DataFrame(df_rows,
                              columns=MultiRegressorPredictor.CLIENT_TRAIN_COLS
                                      + [MultiRegressorPredictor.TARGET_COL])
            y = df[MultiRegressorPredictor.TARGET_COL].as_matrix()
            X = df.drop([MultiRegressorPredictor.TARGET_COL], axis=1).as_matrix()
            dfs[c] = (X, y)

        return dfs

    @staticmethod
    def __prepare_products_dataset(cnx, products_count, from_ts, to_ts):
        # type: (sqlite3.connect, int, long, long) -> dict
        """
        Crea il dataset per il training dei regressori associati ai prodotti, recuperando i dati dalla
        connessione al database e per l'intervallo di tempo specificato dai due timestamp.
        :param cnx: connessione al database da utilizzare
        :param products_count: numero di prodotti presenti
        :param from_ts: timestamp relativo al giorno di partenza del dataset
        :param to_ts: timestamp relativo all'ultimo giorno (incluso) del dataset
        :return: {(X, y)} dizionario indicizzato per id del prodotto contenente le coppie (matrice 
         di train, valori associati) da utilizzare per effettuare il train del relativo regressore
        """
        dfs = {}

        for p in range(0, products_count):
            # Crea il DataFrame con anche le righe per i prodotti non ordinati
            query = "select datetime, count(*) " \
                    "from orders " \
                    "where datetime >= %d and datetime <= %d and product_id = %d " \
                    "group by datetime, product_id " \
                    "order by datetime, client_id, product_id" % (from_ts, to_ts, p)
            Log.d(TAG, query)
            # ^ ORDER BY è fondamentale per effettuare la creazione in modo efficiente
            cursor = cnx.execute(query)

            next_row = cursor.fetchone()
            df_rows = []
            for ts in range(from_ts, to_ts + SECS_IN_DAY, SECS_IN_DAY):
                ordered = 0
                if next_row is not None and next_row[0] == ts:
                    ordered = next_row[1]
                    next_row = cursor.fetchone()

                order_date = datetime.fromtimestamp(ts)
                day_of_year = order_date.timetuple().tm_yday
                year = order_date.timetuple().tm_year
                df_rows.append({
                    'datetime': ts,  # timestamp della data dell'ordine
                    'day_of_year': day_of_year,
                    'year': year,
                    'product_id': p,
                    MultiRegressorPredictor.TARGET_COL: ordered
                })
            df = pd.DataFrame(df_rows,
                              columns=MultiRegressorPredictor.PRODUCT_TRAIN_COLS
                                      + [MultiRegressorPredictor.TARGET_COL])
            y = df[MultiRegressorPredictor.TARGET_COL].as_matrix()
            X = df.drop([MultiRegressorPredictor.TARGET_COL], axis=1).as_matrix()
            dfs[p] = (X, y)
        return dfs

    @staticmethod
    def __prepare_period_dataset(cnx, from_ts, to_ts):
        # type: (sqlite3.connect, long, long) -> (np.ndarray, np.ndarray)
        """
        Crea il dataset per il training del regressore del periodo, recuperando i dati dalla
        connessione al database e per l'intervallo di tempo specificato dai due timestamp.
        :param cnx: connessione al database da utilizzare
        :param from_ts: timestamp relativo al giorno di partenza del dataset
        :param to_ts: timestamp relativo all'ultimo giorno (incluso) del dataset
        :return: (X, y) matrice delle istanze di train e vettore con i valori associati ad ogni istanza
        """

        # Crea il DataFrame con anche le righe per i prodotti non ordinati
        query = "select datetime, count(*) " \
                "from orders " \
                "where datetime >= %d and datetime <= %d " \
                "group by datetime " \
                "order by datetime" % (from_ts, to_ts)
        Log.d(TAG, query)
        # ^ ORDER BY è fondamentale per effettuare la creazione in modo efficiente
        cursor = cnx.execute(query)

        next_row = cursor.fetchone()
        df_rows = []
        for ts in range(from_ts, to_ts + SECS_IN_DAY, SECS_IN_DAY):
            ordered = 0
            if next_row is not None and next_row[0] == ts:
                ordered = next_row[1]
                next_row = cursor.fetchone()

            order_date = datetime.fromtimestamp(ts)
            day_of_year = order_date.timetuple().tm_yday
            year = order_date.timetuple().tm_year
            df_rows.append({
                'datetime': ts,  # timestamp della data dell'ordine
                'day_of_year': day_of_year,
                'year': year,
                MultiRegressorPredictor.TARGET_COL: ordered
            })
        df = pd.DataFrame(df_rows,
                          columns=MultiRegressorPredictor.PERIOD_TRAIN_COLS
                                  + [MultiRegressorPredictor.TARGET_COL])
        y = df[MultiRegressorPredictor.TARGET_COL].as_matrix()
        X = df.drop([MultiRegressorPredictor.TARGET_COL], axis=1).as_matrix()

        return X, y

    def fit(self, cnx, clients_count, products_count, from_ts, to_ts):
        # type: (sqlite3.connect, int, int, long, long) -> None
        """
        Inizializza il predittore recuperando i dati dalla connessione al database       
        :param cnx: connessione al database
        :param clients_count: numero di clienti
        :param products_count: numero di prodotti
        :param from_ts: timestamp del giorno d'inizio dei dati di train
        :param to_ts: timestamp del giorno finale dei dati di train (incluso)
        :return: -
        """
        matrix_shape = (clients_count, products_count)
        self.pcp_estimation = np.zeros(shape=matrix_shape)
        self.product_regressors = np.ndarray(shape=(products_count,), dtype=object)
        self.client_regressors = np.ndarray(shape=(clients_count,), dtype=object)

        days_count = (to_ts - from_ts + SECS_IN_DAY) / SECS_IN_DAY

        # Calcola il numero medio di 1 per ogni cella della matrice
        total_cnt = 0
        query = "select client_id, product_id, count(*) " \
                "from orders " \
                "where datetime >= %d and datetime <= %d " \
                "group by client_id, product_id" % (from_ts, to_ts)
        Log.d(TAG, query)
        for c, p, cnt in cnx.execute(query):
            total_cnt += cnt
            self.pcp_estimation[c, p] = float(cnt) / float(days_count)

        avg = float(total_cnt) / float(days_count)
        avg = int(math.ceil(avg))
        self.avg_ones = 1 if avg == 0 else avg

        # Dati di train per i regressori dei clienti
        Log.d(TAG, "Fit dei regressori per i clienti...")
        clients_train_data = self.__prepare_clients_dataset(cnx,
                                                            clients_count=clients_count,
                                                            from_ts=from_ts,
                                                            to_ts=to_ts)
        for client_id in clients_train_data.keys():
            # NOTA: l'id del cliente deve corrispondere con la posizione della matrice
            X, y = clients_train_data[client_id]
            self.client_regressors[client_id] = self.__build_clf()
            self.client_regressors[client_id].fit(X, y)

        # Dati di train per i regressori dei prodotti
        Log.d(TAG, "Fit dei regressori per i prodotti...")
        products_train_data = self.__prepare_products_dataset(cnx,
                                                              products_count=products_count,
                                                              from_ts=from_ts,
                                                              to_ts=to_ts)
        for product_id in products_train_data.keys():
            # NOTA: l'id del prodotto deve corrispondere con la posizione della matrice
            X, y = products_train_data[product_id]
            self.product_regressors[product_id] = self.__build_clf()
            self.product_regressors[product_id].fit(X, y)

        # Dati di train per il regressore del periodo
        Log.d(TAG, "Fit del regressore del periodo...")
        X_period, y_period = self.__prepare_period_dataset(cnx, from_ts=from_ts, to_ts=to_ts)
        self.period_regressor = self.__build_clf()
        self.period_regressor.fit(X_period, y_period)
        return

    def __calculate_weight(self, c, p, timestamp):
        # type: (int, int, long) -> float
        """
        Calcola il peso della cella [c,p] nella matrice per il giorno rappresentato dal timestamp
        :param c: (int) posizione del cliente nella matrice (coincide con l'id)
        :param p: (int) posizione del prodotto nella matrice (coincide con l'id) 
        :param t: (long) timestamp dell'ordine
        :return: peso della cella
        """

        order_date = datetime.fromtimestamp(timestamp)
        day_of_year = order_date.timetuple().tm_yday
        year = order_date.timetuple().tm_year

        X_client = pd.DataFrame([{
            'datetime': timestamp,  # timestamp della data dell'ordine
            'day_of_year': day_of_year,
            'year': year,
            'client_id': c
        }], columns=MultiRegressorPredictor.CLIENT_TRAIN_COLS).as_matrix()

        X_product = pd.DataFrame([{
            'datetime': timestamp,  # timestamp della data dell'ordine
            'day_of_year': day_of_year,
            'year': year,
            'product_id': p
        }], columns=MultiRegressorPredictor.PRODUCT_TRAIN_COLS).as_matrix()

        X_period = pd.DataFrame([{
            'datetime': timestamp,  # timestamp della data dell'ordine
            'day_of_year': day_of_year,
            'year': year,
        }], columns=MultiRegressorPredictor.PERIOD_TRAIN_COLS).as_matrix()

        weight = 1
        w_c = self.client_regressors[c].predict(X_client)[0]
        w_p = self.product_regressors[p].predict(X_product)[0]
        w_t = self.period_regressor.predict(X_period)[0]

        p_cp = self.pcp_estimation[c, p]

        if 'w_c' in self.components:
            weight *= w_c
        if 'w_p' in self.components:
            weight *= w_p
        if 'w_t' in self.components:
            weight *= w_t
        if 'w_cp' in self.components:
            weight *= p_cp
        # print c, p, w_c, w_p, w_t, prob
        return weight

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

        predictions = np.zeros(shape=self.pcp_estimation.shape, dtype=int)
        clients_count = predictions.shape[0]
        products_count = predictions.shape[1]

        # check threshold
        for c in range(0, clients_count):
            for p in range(0, products_count):
                weight = self.__calculate_weight(c, p, order_timestamp)
                predictions[c, p] = 1 if weight >= threshold else 0
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

        weights = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                weights[c, p] = self.__calculate_weight(c, p, order_timestamp)

        # copy is needed because reasons... (reshape + sort)
        weights_vec = np.reshape(weights.copy(), weights.size)
        weights_vec[::-1].sort()  # in place revese sort

        threshold = weights_vec[self.avg_ones - 1]
        # check threshold
        for c in range(0, clients_count):
            for p in range(0, products_count):
                predictions[c, p] = 1 if weights[c, p] >= threshold else 0

        return predictions

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
                probs[c, p] = self.__calculate_weight(c, p, order_timestamp)

        vectorized_weights = np.reshape(probs, (1, probs.size))  # (1x Nweights)
        result = np.ones(shape=(probs.size, 2))
        result[:, 0] = result[:, 0] - vectorized_weights
        result[:, 1] = vectorized_weights
        return result
