# -*- coding: utf-8 -*-
import datetime
import math
import random
import numpy as np
import pandas as pd

from util import weighted_choice


# Modelli di probabilità per gli ordini
# 1. random: P(c,p,t) = random
# 2. simple: P(c,p,y) = p_c * p_p * p_t
# 3. cond:   P(c,p,y) = P(c,p) * P(c,t) * P(p,t) * p_t
#######

class ProbabilityModel(object):
    """
    Classe base per un modello di probabilità.
    Definsice un metodo `probability` che deve essere implementato da tutte le sotto-classi e un metodo
    `will_make_order` che ritorna 1 o 0 secondo una scelta pesata influenza dal valore calcolato da `probability`.
    """

    def probability(self, client, product, timestamp):
        # type: (pd.Series, pd.Series, long) -> float
        """
        Probabilità che venga effettautato un ordine
        :param client: (np.Series) array che rappresenta un cliente
        :param product: (np.Series) array che rappresenta un prodotto
        :param timestamp: (long) timestamp del giorno dell'ordine
        :return: (float) probabilità dell'ordine
        """
        return NotImplemented

    def will_make_order(self, client, product, timestamp):
        # type: (pd.Series, pd.Series, long) -> int
        """
        Ritorna 1 se il cliente effettuarà un ordine del prodotto il giorno stabilito.
        :param client: (np.Series) array che rappresenta un cliente
        :param product: (np.Series) array che rappresenta un prodotto
        :param timestamp: (long) timestamp del giorno dell'ordine
        :return: (float) probabilità dell'ordine
        """
        p = self.probability(client, product, timestamp)
        choices = [(1, p), (0, 1 - p)]
        return weighted_choice(choices)

    @staticmethod
    def period_probability(t, freq_scale=1):
        # type: (int, float) -> float
        """
        Sinuoide che approssima le periodicità del modello.
        :param t: istante
        :param freq_scale: scala della frequenza della sinusoide. Default: periodo annuo.
        :return: Valore della sinuoide all'istante `t`
        """
        t %= 365  # t >= 0
        t_radiants = 2 * math.pi * float(t) / 364.0  # Converte t in radianti, 2pi = 0 = 365
        prob = (math.sin(t_radiants * freq_scale) + 1) / 2  # Porta la sinuoside nell'intervallo [0,1]
        return prob


###############################


class RandomProbabilityModel(ProbabilityModel):
    """
    Modello di probabilità causale. La probabilità di effettuare un'ordine segue la distribuzione uniforme.
    """

    def probability(self, client, product, timestamp):
        # type: (ProbabilityModel, pd.Series, pd.Series, long) -> float
        return random.uniform(0, 1)


class SimpleProbabilityModel(ProbabilityModel):
    """
    Modello di probabilità semplice. La probabilità di effettuare un'ordine dipende dalla probabilità che ha un
    cliente di effettuare un'ordine (p_c), da quella di un prodotto di essere ordinato (p_p) e dal periodo in cui
    viene effettuato l'ordine (p_t)
    """

    def probability(self, client, product, timestamp):
        # type: (pd.Series, pd.Series, long) -> float
        """
        P(c,p,t) = p_c * p_p * p_t
        :param client: (pd.Series) array rappresentante il cliente
        :param product: (pd.Series) array rappresentate in prodotto
        :param timestamp: (long) timestamp del giorno dell'ordine
        :return: probabilità che venga effettuato un ordine
        """
        order_date = datetime.datetime.fromtimestamp(timestamp)
        t = order_date.timetuple().tm_yday  # Giorno dell'anno

        p_t = ProbabilityModel.period_probability(t)
        p_c = client['pc']
        p_p = product['pp']
        p = p_t * p_p * p_c

        assert p >= 0
        return p


class CondProbabilityModel(ProbabilityModel):
    """
    Modello di probabilità condizionato.
    La probabilità che venga effettuato un'ordine dipende da:
    - P(c,t): La probabilità che un cliente `c` effettui un'ordine nel periodo `t`
    - P(p,t): La probabilità che un prodotto `p` sia ordinato nel periodo `t`
    - P(t): La probabilità che venga effettuato un'ordine nel periodo `t`
    - P(c,p): La probabilità che il cliente `c` ordini il prodotto `p`
    """

    def __init__(self, clients, products):
        # type: (pd.DataFrame, pd.DataFrame) -> None
        clients_count = clients.shape[0]
        products_count = products.shape[0]

        self.__p_cp = np.random.rand(clients_count, products_count)
        # __p_co è la matrice contenente la probabilità cliente-prodotto, la quale modella il fatto che alcuni clienti
        # preferiscono determinati prodotti.

    def probability(self, client, product, timestamp):
        # type: (pd.Series, pd.Series, long) -> float
        """
        P(c,p,t) = P(c,t) * P(p,t) * P(c,p) * p_t con P(c,t) = p_c * (sin(t_c * t) +1)/2
        :param client: (pd.Series) array rappresentante il cliente
        :param product: (pd.Series) array rappresentante il prodotto
        :param timestamp: (long) timestamp della data dell'ordine
        :return: Probabilità che venga effettauto un ordine
        """
        order_date = datetime.datetime.fromtimestamp(timestamp)
        t = order_date.timetuple().tm_yday  # Day of the year

        p_t = ProbabilityModel.period_probability(t)
        p_c = client['pc'] * ProbabilityModel.period_probability(t, client['client_freq_scale'])
        p_p = product['pp'] * ProbabilityModel.period_probability(t, product['product_freq_scale'])

        p = p_t * p_p * p_c * self.__p_cp[int(client['clientId']), int(product['productId'])]

        assert p >= 0
        return p


class NoisyProbabilityModel(ProbabilityModel):
    """
    Modello di probabilità condizionato con l'aggiunta di rumore casuale.
    La probabilità che venga effettuato un'ordine dipende da:
    - P(c,t): La probabilità che un cliente `c` effettui un'ordine nel periodo `t`
    - P(p,t): La probabilità che un prodotto `p` sia ordinato nel periodo `t`
    - P(t): La probabilità che venga effettuato un'ordine nel periodo `t`
    - P(c,p): La probabilità che il cliente `c` ordini il prodotto `p`
    """

    def __init__(self, clients, products):
        # type: (pd.DataFrame, pd.DataFrame) -> None
        clients_count = clients.shape[0]
        products_count = products.shape[0]
        random.seed(clients_count * products_count)
        self.random_state = random.getstate()
        self.__p_cp = np.random.rand(clients_count, products_count)
        # __p_cp è la matrice contenente la probabilità cliente-prodotto, la quale modella il fatto che alcuni clienti
        # preferiscono determinati prodotti.

    def probability(self, client, product, timestamp):
        # type: (pd.Series, pd.Series, long) -> float
        """
        P(c,p,t) = P(c,t) * P(p,t) * P(c,p) * p_t con P(c,t) = p_c * (sin(t_c * t) +1)/2
        :param client: (pd.Series) array rappresentante il cliente
        :param product: (pd.Series) array rappresentante il prodotto
        :param timestamp: (long) timestamp della data dell'ordine
        :return: Probabilità che venga effettauto un ordine
        """
        order_date = datetime.datetime.fromtimestamp(timestamp)
        t = order_date.timetuple().tm_yday  # Day of the year

        p_t = ProbabilityModel.period_probability(t)
        p_c = client['pc'] * ProbabilityModel.period_probability(t, client['client_freq_scale'])
        p_p = product['pp'] * ProbabilityModel.period_probability(t, product['product_freq_scale'])

        p = p_t * p_p * p_c * self.__p_cp[int(client['clientId']), int(product['productId'])]

        # Aggiungo del rumore
        random.setstate(self.random_state)
        noise = random.uniform(0, 1)
        p *= noise
        self.random_state = random.getstate()
        assert p >= 0
        return p
