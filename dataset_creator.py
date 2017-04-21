#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getopt
import json


import sys

import time

from util.file_helper import save_all, load_all
from generator.generators import generate_dataset

J_PREFIX = "prefix"
J_CLIENTS_COUNT = "clients_count"
J_PRODUCTS_COUNT = "products_count"
J_DAYS_COUNT = "days_count"
J_DAY_INTERVAL = "day_interval"
J_MODEL_NAME = "model_name"

# TODO: quantities


def main(argv):
    json_file_path = ""
    try:
        (opts, args) = getopt.getopt(argv, "hf:")
    except getopt.GetoptError, e:
        print "Wrong options"
        print e
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            json_file_path = arg
        if opt in ("-h", "--help"):
            sys.exit()

    if json_file_path == "":
        print "No file."
        sys.exit()

    print "Loading configuration from: ", json_file_path
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
    print json_data

    base_prefix = json_data[J_PREFIX]
    clients_count = json_data[J_CLIENTS_COUNT]
    products_count = json_data[J_PRODUCTS_COUNT]
    days_count = json_data[J_DAYS_COUNT]
    day_interval = json_data[J_DAY_INTERVAL]  # continuous
    model_name = json_data[J_MODEL_NAME]  # 'random'

    # places it in the base_prefix data subdir.
    prefix = '%s_c%d_p%d_d%d_int%d_pmod_%s_' % ( base_prefix, clients_count, products_count, days_count, day_interval, model_name)
    print prefix

    clients, products, orders, model = generate_dataset(clients_count=clients_count,
                                                        products_count=products_count,
                                                        days_count=days_count,
                                                        day_interval=day_interval,
                                                        model_name=model_name)
    save_all(clients, products, orders, model, base_prefix, prefix)


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
#################################################################################################
#
#clients_count = 10
#products_count = 100
#days_count = 10
#day_interval = 0  # continuous
#model_name = 'cond'  # 'random'
#
#
#
#
##################################################################################################
#
#clients_count = 10
#products_count = 100
#days_count = 100
#day_interval = 4
#model_name = 'cond'  # 'random'
#
#prefix = 'c%d_p%d_d%d_int%d_pmod_%s_' % (clients_count, products_count, days_count, day_interval, model_name)
#print prefix
#
#clients, products, orders, model = generate_dataset(clients_count=clients_count,
#                                                    products_count=products_count,
#                                                    days_count=days_count,
#                                                    day_interval=day_interval,
#                                                    model_name=model_name)
#save_all(clients, products, orders, model, prefix)
#
#
##################################################################################################
#
#clients_count = 1000
#products_count = 1000
#days_count = 365
#day_interval = 0  # continuous
#model_name = 'cond'  # 'random'

#prefix = 'c%d_p%d_d%d_int%d_pmod_%s_' % (clients_count, products_count, days_count, day_interval, model_name)
#print prefix
#
#clients, products, orders, model = generate_dataset(clients_count=clients_count,
#                                                    products_count=products_count,
#                                                    days_count=days_count,
#                                                    day_interval=day_interval,
#                                                    model_name=model_name)
#save_all(clients,# products, orders, model, prefix)
#
#
##################################################################################################
#
#clients_count = 1000
#products_count = 1000
#days_count = 365
#day_interval = 4  # continuous
#model_name = 'cond'  # 'random'
#
#prefix = 'c%d_p%d_d%d_int%d_pmod_%s_' % (clients_count, products_count, days_count, day_interval, model_name)
#print prefix
#
#clients, products, orders, model = generate_dataset(clients_count=clients_count,
#                                                    products_count=products_count,
#                                                    days_count=days_count,
#                                                    day_interval=day_interval,
#                                                    model_name=model_name)
#save_all(clients, products, orders, model, prefix)
#