import numpy as np
import time
import math

SECS_IN_DAY = 60*60*24


def scale_fn(t):
    return 1 / t
    # lt = math.log(t,10)
    # if lt != 0:
    #     return 1/lt
    # return 1


class BaselinePredictor(object):
    def __init__(self, scale_function=scale_fn):
        self.scale_function = scale_function
        self.weights = None  # class "1" probability
        self.avg_ones = None

    def fit(self, matrices):
        sample = matrices[matrices.keys()[0]]
        predictions = np.zeros(shape=sample.shape)

        clients_count = sample.shape[0]
        products_count = sample.shape[1]

        # matrices since yesterday (t = 1), dict indexed by date
        # predictions for today (t = 0)
        norm_factor = 0
        for day in matrices.keys():
            t = (time.time() - day) % SECS_IN_DAY
            norm_factor += 1 * scale_fn(t)
            for c in range(0, clients_count):
                for p in range(0, products_count):
                    predictions[c, p] += matrices[day][c, p] * scale_fn(t)

        # print "Calculated values:"
        # print predictions

        # normalize the values
        for c in range(0, clients_count):
            for p in range(0, products_count):
                predictions[c, p] /= norm_factor

        self.weights = predictions

        # calculates the average
        ones_cnt = 0
        for day in matrices.keys():
            matrix = matrices[day]
            ones_cnt += matrix.sum()
        avg = float(ones_cnt) / float(len(matrices.keys()))
        # print 'Float avg', avg
        avg = int(math.ceil(avg))
        # print 'Ceiled avg', avg
        self.avg_ones = 1 if avg == 0 else avg
        return

    def predict_with_threshold(self, threshold):
        """
        Predicts a 1 if the raw predicted value is greater than the given threshold
        :param threshold:
        :return:
        """
        predictions = self.weights.copy()
        clients_count = predictions.shape[0]
        products_count = predictions.shape[1]

        # check threshold
        for c in range(0, clients_count):
            for p in range(0, products_count):
                predictions[c, p] = 1 if predictions[c, p] >= threshold else 0

        return predictions

    def predict_with_topn(self):
        """
        First it calculates the average number of orders in a day and then predicts (approximately) the average number
        of orders by predicting a 1 only for the top-N raw predicted values
        :return:
        """
        clients_count = self.weights.shape[0]
        products_count = self.weights.shape[1]

        # copy is needed because reasons... (reshape + sort)
        predictions_vector = np.reshape(self.weights.copy(), self.weights.size)
        predictions_vector[::-1].sort()  # in place revese sort

        threshold = predictions_vector[self.avg_ones - 1]
        predictions = self.weights.copy()
        print 'Threshold', threshold
        # check threshold
        for c in range(0, clients_count):
            for p in range(0, products_count):
                predictions[c, p] = 1 if predictions[c, p] >= threshold else 0

        return predictions

    def predict_proba(self):
        vectorized_weights = np.reshape(self.weights, (1, self.weights.size)) # (1x Nweights)
        result = np.ones(shape=(self.weights.size, 2))
        result[:, 0] = result[:, 0] - vectorized_weights
        result[:, 1] = vectorized_weights
        return result
