# -*- coding: utf-8 -*-
import numpy as np
import math

SECS_IN_DAY = 60 * 60 * 24


def scale_fn(t):
    # type: (int) -> float
    return 1.0 / t


class BaselinePredictor(object):
    """
    Predittore base che effettua la predizione creando un matrice `c x p` le cui celle contengono il numero medio
    di ordini effettuati dalla coppia (c,p), scalati in funzione del tempo trascorso.
    """

    def __init__(self, scale_function=scale_fn):
        # type: (callable) -> None
        self.scale_function = scale_function
        self.matrices = {}  # type: dict - dizionario contenente le matrici degli ordini
        self.avg_ones = None  # type: float

    def __calculate_weights(self, timestamp):
        # type: (long) -> np.ndarray
        """
        Calcola la matrice dei pesi scalando ogni peso in base al tempo trascorso
        :param timestamp: (long) data di riferiemento dell'ordine
        """
        sample = self.matrices[self.matrices.keys()[0]]
        clients_count = sample.shape[0]
        products_count = sample.shape[1]

        weights = np.zeros(shape=(clients_count, products_count))

        norm_factor = 0
        for day in self.matrices.keys():
            t = int((timestamp - day) /SECS_IN_DAY)
            norm_factor += 1 * scale_fn(t)
            for c in range(0, clients_count):
                for p in range(0, products_count):
                    weights[c, p] += self.matrices[day][c, p] * scale_fn(t)

        # Normalizza i valori nell'intervallo [0,1]
        for c in range(0, clients_count):
            for p in range(0, products_count):
                weights[c, p] /= norm_factor
        return weights

    def fit(self, matrices):
        # type: (dict) -> None
        """
        Inizializza il predittore
        :param matrices: dizionario di matrici degli ordini
        :return:
        """
        self.matrices = matrices
        # Calcola il numero giornaliero di ordini medio
        ones_cnt = 0
        for day in matrices.keys():
            matrix = matrices[day]
            ones_cnt += matrix.sum()
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
        if len(self.matrices.keys()) <= 0:
            return None
        clients_count = self.matrices[self.matrices.keys()[0]].shape[0]
        products_count = self.matrices[self.matrices.keys()[0]].shape[1]
        predictions = np.zeros(shape=(clients_count, products_count))

        weights = self.__calculate_weights(order_timestamp)

        for c in range(0, clients_count):
            for p in range(0, products_count):
                predictions[c, p] = 1 if weights[c, p] >= threshold else 0

        return predictions

    def predict_with_topn(self, order_timestamp):
        # type: (long) -> np.ndarray or None
        """
        Utilizza come threshold per le predizioni un valore tale che vengono predetti tanti ordini quanto è il numero
        medio di orgini che viene effettuato giornalmente
        :param order_timestamp: (long) timestamp della data dell'ordine
        :return: (np.ndarray) matrice degli ordini relativa al timestamp
        """
        if len(self.matrices.keys()) <= 0:
            return None

        clients_count = self.matrices[self.matrices.keys()[0]].shape[0]
        products_count = self.matrices[self.matrices.keys()[0]].shape[1]
        predictions = np.zeros(shape=(clients_count, products_count))

        weights = self.__calculate_weights(order_timestamp)

        # Serve la copia perché numpy sfasa un po' di cose... (reshape + sort)
        predictions_vector = np.reshape(weights.copy(), weights.size)
        predictions_vector[::-1].sort()  # Magicamente fa l'inplace reverse-sort

        threshold = predictions_vector[self.avg_ones - 1]

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
        if len(self.matrices.keys()) <= 0:
            return None
        weights = self.__calculate_weights(order_timestamp)

        vectorized_weights = np.reshape(weights, (1, weights.size))  # (1x N_weights)
        result = np.ones(shape=(weights.size, 2))
        result[:, 0] = result[:, 0] - vectorized_weights
        result[:, 1] = vectorized_weights
        return result
