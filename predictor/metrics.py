import numpy as np


def calculate_metrics(predictions, expected):
    # type: (np.ndarray, np.ndarray) -> (float, float, float)
    """Calcola accuracy, precision and recall confrontando la matrice predetta con quella dei risultati attesi
    """
    clients_count = predictions.shape[0]
    products_count = predictions.shape[1]

    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0

    total = float(clients_count * products_count)

    for c in range(0, clients_count):
        for p in range(0, products_count):
            if predictions[c, p] == expected[c, p]:
                if predictions[c, p] == 1:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if predictions[c, p] == 1:
                    false_positive += 1
                else:
                    false_negative += 1

    accuracy = float(true_positive + true_negative) / total
    if true_positive + false_positive == 0:
        precision = 0
    else:
        precision = true_positive / float(true_positive + false_positive)

    if true_positive + false_negative == 0:
        recall = 0
    else:
        recall = true_positive / float(true_positive + false_negative)

    return accuracy, precision, recall
