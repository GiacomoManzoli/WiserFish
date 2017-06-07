#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time

import datetime
import numpy as np
from sklearn.metrics import roc_auc_score

from dataset.generator.generator_v2 import generate_dataset
from dataset.generator.polynomial import Polynomial, PeriodicPolynomial, PeriodicThresholdFunction
from dataset.generator.product import Product
from dataset.generator.values import *
from dataset.manager import save_dataset, load_dataset, make_order_matrices


def main(argv):
    # poly = PeriodicPolynomial(grade=20, theta=1)
    # poly.plot(range_min=-1, range_max=1)

    poly2 = PeriodicPolynomial(degree=20, theta=1, build_mode='fit', points=15)
    # poly2.plot(range_min=-1, range_max=1)

    points = [
        (0, 1),
        (1, 1.1),
        (2, 0.4),
        (3, 1.9),
        (4, 1),
        (5, 2),
        (6, 1)
    ]

    fn = PeriodicThresholdFunction.create_from_points(points)
    # fn.plot()

    for p in points:
        x, y = p
        print x, y, x / 6.0, fn(x / 6.0)
        assert fn(x / 6.0) == y

    # products = [Product(i) for i in range(0,10)]

    # client = ClientFactory.create_consumption_client(id=0,
    #                                                  products=products,
    #                                                  client_type=CLIENT_TYPE_GDO,
    #                                                  high_consumption_ratio=0.25,
    #                                                  no_consumption_ratio=0.5)
    # client = ClientFactory.create_regular_client(id=0,
    #                                              products=products,
    #                                              client_type=CLIENT_TYPE_GDO,
    #                                              order_days=[MONDAY, FRIDAY],
    #                                              high_consumption_ratio=0.25,
    #                                              no_consumption_ratio=0.5)
    # client = ClientFactory.create_random_client(id=0,
    #                                             products=products,
    #                                             client_type=CLIENT_TYPE_GDO,
    #                                             high_consumption_ratio=0.25,
    #                                             no_consumption_ratio=0.5)
    # global_trend = PeriodicPolynomial(degree=20, theta=1, build_mode='fit', points=20)

    # global_trend.plot(0, 1)

    clients, products, orders, global_trend = generate_dataset(product_count=10,
                                                               seasonal_ratio=0.8,
                                                               client_count=4,
                                                               resturant_ratio=0.25,
                                                               gdo_ratio=0.5, regular_ratio=0.2,
                                                               consumption_ratio=0.6,
                                                               from_ts=time.time(),
                                                               to_ts=time.time() + 15 * SECS_IN_DAY)
    # print "--- Products ---"
    # for p in products:
    #    print p
    #
    # print "--- Client ---"
    # client = clients[0]
    # print client
    # print client.consumptions
    #
    # print "--- Orders ---"
    # for o in orders:
    #    print o

    save_dataset(clients=clients,
                 products=products,
                 orders=orders,
                 global_trend=global_trend,
                 dataset_name='sample')
    client_df, product_df, order_df = load_dataset('sample')

    matrices = make_order_matrices(len(clients), len(products), order_df)

    print matrices


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
