import datetime
import numpy as np
import math

from generator.probability_models import ProbabilityModel

SECS_IN_DAY = 60 * 60 * 24


class LessSinfulBaselinePredictor(object):
    """
    Predictor which uses only the period sinusoid form the probability model for the data generation instead
    of all the model (hence LessSinful).
    P_p, P_c and P_cp are approximated using the train set (with the average).
    """

    def __init__(self):
        self.avg_ones = None
        self.pp_estimation = None  # Estimation of P(p)
        self.pc_estimation = None  # Estimation of P(c)
        self.pcp_estimation = None  # Estimation of P(cp)

    def fit(self, matrices):
        sample = matrices[matrices.keys()[0]]
        self.pcp_estimation = np.zeros(shape=sample.shape)

        clients_count = sample.shape[0]
        products_count = sample.shape[1]

        self.pp_estimation = np.zeros(shape=(products_count,))
        self.pc_estimation = np.zeros(shape=(clients_count,))

        ones_cnt = 0  # counts the total number of orders
        days_count = len(matrices.keys())
        for day in matrices.keys():
            ones_cnt += matrices[day].sum()
            for c in range(0, clients_count):
                for p in range(0, products_count):
                    self.pcp_estimation[c, p] += matrices[day][c, p] / days_count
                    self.pp_estimation[p] += matrices[day][c, p] / days_count
                    self.pc_estimation[c] += matrices[day][c, p] / days_count
        # estimations[c, p] = # of times that the client c bought the product p
        # ^ estimate of p(c,p)
        # pp_estimation[p] = # of times that the product p has been bought
        # ^ estimate of p(p)
        # pc_estimation[p] = # of times that the client c acquired a product

        # calculates the average number of orders in a day
        avg = float(ones_cnt) / float(len(matrices.keys()))
        avg = int(math.ceil(avg))
        self.avg_ones = 1 if avg == 0 else avg
        return

    def __calculate_order_probability(self, c, p, t):
        p_t = ProbabilityModel.period_probability(t)
        p_c = self.pc_estimation[c]
        p_p = self.pp_estimation[p]
        p_cp = self.pcp_estimation[c, p]
        return p_c * p_p * p_cp * p_t

    def predict_with_threshold(self, order_timestamp, threshold):
        """
        Predicts a 1 if the probability predicted value is greater than the given threshold
        :param order_timestamp:
        :param threshold:
        :return:
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
        """
        First it calculates the average number of orders in a day and then predicts (approximately) the average number
        of orders by predicting a 1 only for the top-N raw predicted values
        :return:
        """
        if self.pcp_estimation is None:
            return None
        clients_count = self.pcp_estimation.shape[0]
        products_count = self.pcp_estimation.shape[1]

        order_date = datetime.datetime.fromtimestamp(order_timestamp)
        # print order_date.strftime('%Y-%m-%d %H:%M:%S')
        t = order_date.timetuple().tm_yday  # Day of the year

        probs = np.zeros(shape=(clients_count, products_count))
        for c in range(0, clients_count):
            for p in range(0, products_count):
                probs[c,p] = self.__calculate_order_probability(c, p, t)

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
