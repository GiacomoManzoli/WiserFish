import numpy as np
import time

SECS_IN_DAY = 60*60*24


def baseline_predictor(matrices, threshold):
    sample = matrices[matrices.keys()[0]]
    predictions = np.zeros(shape=sample.shape)

    clients_count = sample.shape[0]
    products_count = sample.shape[1]

    def scale_fn(t):
        return 1/t
        #lt = math.log(t,10)
        #if lt != 0:
        #    return 1/lt
        #return 1

    # matrices since yesterday (t = 1), dict indexed by date
    # predictions for today (t = 0)
    for day in matrices.keys():
        t = (time.time() - day) % SECS_IN_DAY
        for c in range(0, clients_count):
            for p in range(0, products_count):
                predictions[c, p] += matrices[day][c, p] * scale_fn(t)

    # print "Calculated values:"
    # print predictions

    # check threshold
    for c in range(0, clients_count):
        for p in range(0, products_count):
            predictions[c, p] = 1 if predictions[c, p] >= threshold else 0

    return predictions
