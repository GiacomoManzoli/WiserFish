import time

from predictor import baseline_predictor
from predictor import calculate_metrics
from util.file_helper import save_all, load_all
from generator.generators import generate_orders, generate_clients, generate_products

SECS_IN_DAY = 60*60*24


# TODO: quantities

print "Generating clients..."
clients = generate_clients(100)
# print clients

print "Generating products..."
products = generate_products(1000)

cnt_days = 50
days = [time.time() - SECS_IN_DAY * i for i in range(1, cnt_days+1)]  # from yesterday to cnt_days back

orders = generate_orders(clients, products, days)

predicted = baseline_predictor(orders, 0.0004)
today = time.time()
expected = generate_orders(clients, products, [today])[today]

print "Predicted"
print predicted
print "Expected"
print expected

accuracy, precision, recall = calculate_metrics(predicted, expected)

print "Accuracy:", accuracy
print "Precision:", precision
print "Recall:", recall

save_all(clients, products, orders)
load_all()
