import datetime

import pandas as pd
from sklearn import linear_model

import numpy as np
import math

from generator.probability_models import ProbabilityModel
from predictor.proposizionalizer import proposizionalize

SECS_IN_DAY = int(60 * 60 * 24)


class SingleRegressorPredictor(object):
    """
    Predictor which uses one Regressor for each cell of the matrix to estimate if the there will be an order
    for the pair (c,p)
    """

    def __init__(self, group_size=5):
        self.avg_ones = None
        self.group_size = int(group_size)
        self.regressor_matrix = None
        self.clients = None
        self.products = None

    def __extract_day_group(self, dataset):
        df = dataset
        df['day_group'] = df['datetime'] / (SECS_IN_DAY * self.group_size)
        df['day_group'] = df['day_group'].astype(int)
        df = df.drop('datetime', axis=1)
        df = df.drop('year', axis=1)
        df = df.drop('day_of_year', axis=1)
        return df

    def fit(self, clients, products, orders, matrices):
        self.clients = clients
        self.products = products
        dataset = proposizionalize(orders, clients, products)

        client_cnt = dataset['client_id'].nunique()
        product_cnt = dataset['product_id'].nunique()

        self.regressor_matrix = np.ndarray(shape=(client_cnt, product_cnt), dtype=object)
        df = self.__extract_day_group(dataset)

        for index, group in df.groupby(['client_id', 'product_id']):
            c, p = index
            # print "Fitting the Regressor for client", c, "product", p
            group = group.groupby('day_group').mean()  #
            # df['ordered'] in range [0,1] and it's the estimated probability of an order in the daygroup of size
            # `group_size'
            # All the values are constant insied a group (excepct the ordered), so taking the mean doens't alterate
            # the values
            group = group.reset_index()

            X = group.drop(['ordered'], axis=1).as_matrix()
            y = group['ordered'].as_matrix()

            # print group.head()

            self.regressor_matrix[c, p] = linear_model.SGDRegressor()
            self.regressor_matrix[c, p].fit(X, y)

        ones_cnt = 0  # counts the total number of orders
        for day in matrices.keys():
            ones_cnt += matrices[day].sum()

        avg = float(ones_cnt) / float(len(matrices.keys()))
        avg = int(math.ceil(avg))
        self.avg_ones = 1 if avg == 0 else avg
        return

    def __calculate_order_probability(self, c, p, timestamp):
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
        """
        Predicts a 1 if the probability predicted value is greater than the given threshold
        :param order_timestamp:
        :param threshold:
        :return:
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
        """
        First it calculates the average number of orders in a day and then predicts (approximately) the average number
        of orders by predicting a 1 only for the top-N raw predicted values
        :return:
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
        """
        Retuns the predicted probabilities in a sklearn-like fashion
        :param order_timestamp:
        :return:
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
