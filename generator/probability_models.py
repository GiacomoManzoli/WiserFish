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
# 3. cond:   P(c,p,y) = P(c,p) * P(c,t) * P(p,t) * p_t
#######

class ProbabilityModel(object):
    """
    Base class for a probability model.
    Defines a `probability` method that has to be implemented by the subclasses and a `will_make_order` method
    which return 1 or 0 as a weighted choices using the probabilty returned by the `probability method`
    """

    def probability(self, client, product, timestamp):
        # type: (ProbabilityModel, pd.Series, pd.Series, long) -> float
        """
        Probability of an order
        :param client: (np.Series) array representing the client
        :param product: (np.Series) array representing the product
        :param timestamp: (long) day of the order
        :return: (float) probability of the order
        """
        # type: (ProbabilityModel, pd.Series, pd.Series, int) -> int
        return NotImplemented

    def will_make_order(self, client, product, timestamp):
        p = self.probability(client, product, timestamp)
        choices = [(1, p), (0, 1 - p)]
        return weighted_choice(choices)

    @staticmethod
    def period_probability( t, freq_scale = 1):
        # type: (int, float) -> float
        """ Annual periodicity effect"""
        t %= 365  # t >= 0
        t_radiants = 2 * math.pi * float(t) / 364.0  # maps t into radiants to get a [0,2pi] sin-like 1-year peridodicty
        prob = (math.sin(t_radiants * freq_scale) + 1) / 2  # converts t in radiants
        return prob

###############################


class RandomProbabilityModel(ProbabilityModel):

    def probability(self, client, product, timestamp):
        # type: (ProbabilityModel, pd.Series, pd.Series, long) -> float
        return random.uniform(0, 1)


class SimpleProbabilityModel(ProbabilityModel):
    def probability(self, client, product, timestamp):
        # type: (ProbabilityModel, pd.Series, pd.Series, long) -> float
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
        return p


class CondProbabilityModel(ProbabilityModel):

    def __init__(self, clients, products):
        clients_count = clients.shape[0]
        products_count = products.shape[0]

        self.__p_cp = np.random.rand(clients_count, products_count)
        # __p_cp is the client product probability which model the fact that certain
        # clients likes/prefers more certain products and then the probability which a client
        # orders a products it's higher if he likes it.

    def probability(self, client, product, timestamp):
        # type: (ProbabilityModel, pd.Series, pd.Series, long) -> float
        # client : array representing the client
        # product : array representing the product
        # timestamp : day of the order

        # P(c,p,y) = P(c,t) * P(p,t) * P(c,p) * p_t with P(c,t) = p_c * (sin(t_c * t) +1)/2

        order_date = datetime.datetime.fromtimestamp(timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        p_t = ProbabilityModel.period_probability(t)
        p_c = client['pc'] * ProbabilityModel.period_probability(t, client['client_freq_scale'])
        p_p = product['pp'] * ProbabilityModel.period_probability(t, product['product_freq_scale'])

        p = p_t * p_p * p_c * self.__p_cp[int(client['clientId']), int(product['productId'])]

        assert p >= 0
        return p

