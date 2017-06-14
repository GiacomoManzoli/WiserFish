# -*- coding: utf-8 -*-
import math

import numpy as np
import pandas as pd
from datetime import datetime

import sqlite3
from sklearn import linear_model
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.svm import SVR

from dataset.generator.values import SECS_IN_DAY

from util import Log

TAG = "SingleRegressorPredictor"


class SingleRegressorPredictor(object):
    """
    Predittore che usa un regressore per ogni cella della matrice clienti-prodotti per approssimare
    la probabilità di una vendita al variare del periodo.    
    I regressori possono essere addestrati sui dati giornalieri oppure aggregando i giorni.
    """

    TRAIN_COLS = ['datetime', 'day_of_year', 'year']
    TARGET_COL = 'ordered'

    def __init__(self, regressor_name='SVR'):
        # type: (str) -> None
        self.regressor_name = regressor_name
        self.avg_ones = None
        self.regressor_matrix = None

    def fit(self, cnx, clients_count, products_count, from_ts, to_ts):
        # type: (sqlite3.connect, int, int, long, long) -> None
        """
        Inizializza il predittore recuperando i dati dal database, considerando solamente i dati nell'intervallo
        temporale specificato (estremi inclusi).
        :param cnx: connessione al database
        :param clients_count: numero di clienti
        :param products_count: numero di prodotti
        :param from_ts: timestamp del giorno d'inizio dei dati di train
        :param to_ts: timestamp del giorno finale dei dati di train (incluso)
        :return: 
        """
        self.regressor_matrix = np.ndarray(shape=(clients_count, products_count), dtype=object)

        days_count = (to_ts - from_ts + SECS_IN_DAY) / SECS_IN_DAY
        # Calcola il numero medio di 1 per ogni cella della matrice
        query = "SELECT count(*) " \
                "FROM orders " \
                "WHERE datetime >= %d AND datetime <= %d " % (from_ts, to_ts)
        Log.d(TAG, query)
        row = cnx.execute(query).fetchone()
        total_cnt = row[0]
        avg = float(total_cnt) / float(days_count)
        avg = int(math.ceil(avg))
        self.avg_ones = 1 if avg == 0 else avg

        # df = self.__extract_day_group(dataset)
        Log.d(TAG, "Fit dei regressori...")
        for c in range(0, clients_count):
            for p in range(0, products_count):
                # Crea il DataFrame con anche le righe per i prodotti non ordinati
                query = "SELECT datetime " \
                        "FROM orders " \
                        "WHERE datetime >= %d " \
                        "AND datetime <= %d " \
                        "AND client_id = %d " \
                        "AND product_id = %d " \
                        "ORDER BY datetime" % (from_ts, to_ts, c, p)
                # ^ ORDER BY è fondamentale per effettuare la creazione in modo efficiente
                Log.d(TAG, query)
                cursor = cnx.execute(query)

                next_row = cursor.fetchone()
                df_rows = []
                for ts in range(from_ts, to_ts + SECS_IN_DAY, SECS_IN_DAY):
                    ordered = 0
                    if next_row is not None and next_row[0] == ts:
                        ordered = 1
                        next_row = cursor.fetchone()

                    order_date = datetime.fromtimestamp(ts)
                    day_of_year = order_date.timetuple().tm_yday
                    year = order_date.timetuple().tm_year
                    df_rows.append({
                        'datetime': ts,  # timestamp della data dell'ordine
                        'day_of_year': day_of_year,
                        'year': year,
                        SingleRegressorPredictor.TARGET_COL: ordered
                    })
                df = pd.DataFrame(df_rows,
                                  columns=SingleRegressorPredictor.TRAIN_COLS
                                          + [SingleRegressorPredictor.TARGET_COL])
                y = df[SingleRegressorPredictor.TARGET_COL].as_matrix()
                X = df.drop([SingleRegressorPredictor.TARGET_COL], axis=1).as_matrix()

                clf = None
                if self.regressor_name == 'SGD':
                    clf = linear_model.SGDRegressor()
                elif self.regressor_name == 'SVR':
                    clf = SVR()
                elif self.regressor_name == 'PAR':
                    clf = PassiveAggressiveRegressor()

                self.regressor_matrix[c, p] = clf
                self.regressor_matrix[c, p].fit(X, y)

        return

    def __calculate_weights(self, c, p, timestamp):
        # type: (int, int, long) -> float
        """
        Calcola la probabilità che il cliente c ordini il prodotto p nel periodo t
        :param c: (int) posizione del cliente nella matrice (coincide con l'id)
        :param p: (int) posizione del prodotto nella matrice (coincide con l'id) 
        :param t: (long) timestamp dell'ordine
        :return: probabilità che venga effettuato l'ordine
        """
        order_date = datetime.fromtimestamp(timestamp)
        day_of_year = order_date.timetuple().tm_yday
        year = order_date.timetuple().tm_year

        X = pd.DataFrame([{
            'datetime': timestamp,  # timestamp della data dell'ordine
            'day_of_year': day_of_year,
            'year': year,
        }], columns=SingleRegressorPredictor.TRAIN_COLS).as_matrix()

        return self.regressor_matrix[c, p].predict(X)[0]

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

        predictions = np.zeros(shape=self.regressor_matrix.shape, dtype=int)
        clients_count = predictions.shape[0]
        products_count = predictions.shape[1]

        # check threshold
        for c in range(0, clients_count):
            for p in range(0, products_count):
                order_probability = self.__calculate_weights(c, p, order_timestamp)
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
                probs[c, p] = self.__calculate_weights(c, p, order_timestamp)

        # copy is needed because reasons... (reshape + sort)
        probs_vec = np.reshape(probs.copy(), probs.size)
        probs_vec[::-1].sort()  # in place revese sort

        threshold = probs_vec[self.avg_ones - 1]
        predictions = np.zeros(shape=(clients_count, products_count))
        # check threshold
        for c in range(0, clients_count):
            for p in range(0, products_count):
                order_probability = self.__calculate_weights(c, p, order_timestamp)
                predictions[c, p] = 1 if order_probability >= threshold else 0
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
        if self.regressor_matrix is None:
            return None
        clients_count = self.regressor_matrix.shape[0]
        products_count = self.regressor_matrix.shape[1]

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c, p] = self.__calculate_weights(c, p, order_timestamp)

        vectorized_weights = np.reshape(probs, (1, probs.size))  # (1x Nweights)
        result = np.ones(shape=(probs.size, 2))
        result[:, 0] = result[:, 0] - vectorized_weights
        result[:, 1] = vectorized_weights
        return result
