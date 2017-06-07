# -*- coding: utf-8 -*-
import datetime
import numpy as np
import math

from dataset.generator.probability_models import ProbabilityModel


class LessSinfulBaselinePredictor(object):
    """
    Predittore che usa la stessa sinusoide del modello di probabilità che modella l'andamento periodico degli ordini.
    (per questo è stato chiamato less-sinful)
    P_p, P_c e P_cp sono approssimate prendendo il valore medio del train set
    """

    def __init__(self):
        self.avg_ones = None
        self.pp_estimation = None  # Stima di P(p)
        self.pc_estimation = None  # Stima di P(c)
        self.pcp_estimation = None  # Stima di P(cp)

    def fit(self, matrices):
        # type: (dict) -> None
        """
        Inizializza la classe
        :param matrices: dizionario di matrici degli ordini
        :return:
        """
        sample = matrices[matrices.keys()[0]]
        self.pcp_estimation = np.zeros(shape=sample.shape)

        clients_count = sample.shape[0]
        products_count = sample.shape[1]

        self.pp_estimation = np.zeros(shape=(products_count,))
        self.pc_estimation = np.zeros(shape=(clients_count,))

        ones_cnt = 0
        days_count = len(matrices.keys())
        for day in matrices.keys():
            matrix = matrices[day]
            ones_cnt += matrix.sum()
            # Media delle righe e delle celle
            for c in range(0, clients_count):
                row_avg = matrix[c, :].sum() / products_count  # numero medio di ordini effetuati dal cliente
                self.pc_estimation[c] += row_avg / days_count  # lo aggiungo alla media, dividendo già per il totale
                for p in range(0, products_count):
                    self.pcp_estimation[c, p] += matrix[c, p] / days_count
            # Media delle colonne
            for p in range(0, products_count):
                col_avg = matrix[:, p].sum() / clients_count  # numero medio di volte che il prodotto è stato ordinato
                self.pp_estimation[p] += col_avg / days_count
        # pcp_estimations[c, p] = # medio di volte che il cliente c ha effettuato un ordine del prodotto p
        # ^ stima di p(c,p)
        # pp_estimation[p] = media della media delle volte che è stato ordinato il prodotto p
        # ^ stima di p(p)
        # pc_estimation[c] = media della media di volte che il cliente c ha effettuato un ordine
        # ^ stima di p(c)

        # Calcola il numero giornaliero di ordini medio
        avg = float(ones_cnt) / float(len(matrices.keys()))
        avg = int(math.ceil(avg))
        self.avg_ones = 1 if avg == 0 else avg
        return

    def __calculate_order_probability(self, c, p, t):
        # type: (int, int, int) -> float
        """
        Calcola la probabilità che il cliente c ordini il prodotto p nel periodo t
        :param c: (int) posizione del cliente nella matrice (coincide con l'id)
        :param p: (int) posizione del prodotot nella matrice (coincide con l'id) 
        :param t: (int) periodo dell'anno
        :return: probabilità che venga effettuato l'ordine
        """
        p_t = ProbabilityModel.period_probability(t)
        p_c = self.pc_estimation[c]
        p_p = self.pp_estimation[p]
        p_cp = self.pcp_estimation[c, p]
        return p_c * p_p * p_cp * p_t

    def predict_with_threshold(self, order_timestamp, threshold):
        # type: (long, float) -> np.ndarray or None
        """
        Predice un ordine (1) se la corrispondente probabilità stimata è maggiore del parametro threshold passato
        :param order_timestamp: (long) timestamp della data dell'ordine
        :param threshold: (float) soglia sopra la quale prevedere un 1
        :return: (np.ndarray) matrice degli ordini relativa al timestamp
        """
        if self.pcp_estimation is None:
            return None

        predictions = np.zeros(shape=self.pcp_estimation.shape)
        clients_count = predictions.shape[0]
        products_count = predictions.shape[1]

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        # check threshold
        for c in range(0, clients_count):
            for p in range(0, products_count):
                order_probability = self.__calculate_order_probability(c, p, t)
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
        clients_count = self.pcp_estimation.shape[0]
        products_count = self.pcp_estimation.shape[1]

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        t = order_date.timetuple().tm_yday  # Day of the year

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c, p] = self.__calculate_order_probability(c, p, t)

        # Serve la copia perché numpy sfasa un po' di cose... (reshape + sort)
        probs_vec = np.reshape(probs.copy(), probs.size)
        probs_vec[::-1].sort()  # Magicamente fa l'inplace reverse-sort

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

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        probs = np.zeros(shape=self.pcp_estimation.shape)
        for c in range(0, self.pcp_estimation.shape[0]):
            for p in range(0, self.pcp_estimation.shape[1]):
                probs[c, p] = self.__calculate_order_probability(c, p, t)

        vectorized_weights = np.reshape(probs, (1, probs.size))  # (1x Nweights)
        result = np.ones(shape=(probs.size, 2))
        result[:, 0] = result[:, 0] - vectorized_weights
        result[:, 1] = vectorized_weights
        return result
