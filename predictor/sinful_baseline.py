import datetime
import numpy as np
import math

from generator.probability_models import CondProbabilityModel


class SinfulBaselinePredictor(object):
    """
    Predictor which uses the same probability model used for the generation of the data.
    The prediction is made in a TopN fashion, selecting the average number of orders.
    (The generation aglorithm instead of an average uses a weighted choice)
    """

    def __init__(self):
        self.clients = None
        self.products = None
        self.prob_model = None
        self.avg_ones = None

    def fit(self, clients, products, matrices):
        self.clients = clients
        self.products = products
        self.prob_model = CondProbabilityModel(clients, products)

        ones_cnt = 0  # counts the total number of orders
        for day in matrices.keys():
            ones_cnt += matrices[day].sum()

        avg = float(ones_cnt) / float(len(matrices.keys()))
        avg = int(math.ceil(avg))
        self.avg_ones = 1 if avg == 0 else avg
        return

    def predict_with_threshold(self, order_timestamp, threshold):
        """
        Predicts a 1 if the probability predicted value is greater than the given threshold
        :param order_timestamp:
        :param threshold:
        :return:
        """
        if self.prob_model is None:
            return None

        clients_count = self.clients.shape[0]
        products_count = self.products.shape[0]

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        predictions = np.ndarray(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                order_probability = self.prob_model.probability(self.clients.ix[c], self.products.ix[p], t)
                predictions[c, p] = 1 if order_probability >= threshold else 0
        return predictions

    def predict_with_topn(self, order_timestamp):
        """
        First it calculates the average number of orders in a day and then predicts (approximately) the average number
        of orders by predicting a 1 only for the top-N raw predicted values
        :return:
        """
        if self.prob_model is None:
            return None

        clients_count = self.clients.shape[0]
        products_count = self.products.shape[0]

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c, p] = self.prob_model.probability(self.clients.ix[c], self.products.ix[p], t)

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
        if self.prob_model is None:
            return None

        clients_count = self.clients.shape[0]
        products_count = self.products.shape[0]

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c, p] = self.prob_model.probability(self.clients.ix[c], self.products.ix[p], t)

        vectorized_weights = np.reshape(probs, (1, probs.size))  # (1x Nweights)
        result = np.ones(shape=(probs.size, 2))
        result[:, 0] = result[:, 0] - vectorized_weights
        result[:, 1] = vectorized_weights
        return result
