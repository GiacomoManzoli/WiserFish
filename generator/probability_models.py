import datetime
import math
import random
import numpy as np
import pandas as pd

from util import weighted_choice


#######
# Probability models for the orders
# 1. random: P(c,p,t) = random
# 2. simple: P(c,p,y) = p_c * p_p * p_t
# 3. cond:   P(c,p,y) = P(c,p) * P(c,t) * P(p,t) * p_t with P(c,t) = p_c * TE * cos(t)
#######

class ProbabilityModel(object):
    def probability(self, client, product, timestamp):
        # type: (ProbabilityModel, pd.Series, pd.Series, int) -> int
        return NotImplemented

    @staticmethod
    def period_probability( t, freq_scale = 1):
        # type: (int, float) -> float
        """ Annual periodicity effect"""
        t %= 365  # t >= 0
        t_radiants = 2 * math.pi * float(t) / 364.0  # maps t into radiants to get a [0,2pi] sin-like 1-year peridodicty
        prob = (math.sin(t_radiants * freq_scale) + 1) / 2  # converts t in radiants
        return prob


class RandomProbabilityModel(ProbabilityModel):
    def probability(self, client, product, timestamp):
        # type: (ProbabilityModel, pd.Series, pd.Series, int) -> int
        p = random.uniform(0, 1)
        choices = [(1, p), (0, 1 - p)]
        return weighted_choice(choices)


class SimpleProbabilityModel(ProbabilityModel):
    def probability(self, client, product, timestamp):
        # type: (ProbabilityModel, pd.Series, pd.Series, int) -> int
        # client : array representing the client
        # product : array representing the product
        # timestamp : day of the order

        # P(c,p,y) = p_c * p_p * p_t

        order_date = datetime.datetime.fromtimestamp(timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        p_t = ProbabilityModel.period_probability(t)
        p_c = client['pc']
        p_p = product['pp']
        p = p_t * p_p * p_c

        assert p >= 0
        choices = [(1, p), (0, 1 - p)]
        return weighted_choice(choices)


class CondProbabilityModel(ProbabilityModel):

    def __init__(self, clients, products):
        clients_count = clients.shape[0]
        products_count = products.shape[0]
        self.__p_cp = np.random.rand(clients_count, products_count)

    def probability(self, client, product, timestamp):
        # type: (ProbabilityModel, pd.Series, pd.Series, int) -> int
        # client : array representing the client
        # product : array representing the product
        # timestamp : day of the order

        # P(c,p,y) = P(c,t) * P(p,t) * p_t with P(c,t) = p_c * TE * (sin(t_c * t) +1)/2

        order_date = datetime.datetime.fromtimestamp(timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        p_t = ProbabilityModel.period_probability(t)
        p_c = client['pc'] * ProbabilityModel.period_probability(t, client['client_freq_scale'])
        p_p = product['pp'] * ProbabilityModel.period_probability(t, product['product_freq_scale'])

        p = p_t * p_p * p_c * self.__p_cp[int(client['clientId']), int(product['productId'])]

        assert p >= 0
        choices = [(1, p), (0, 1 - p)]
        return weighted_choice(choices)

