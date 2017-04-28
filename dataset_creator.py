#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getopt
import sys
import time

from util.file_helper import save_all, load_all, save_partial_orders, ConfigurationFile
from generator.generators import generate_dataset, generate_orders, generate_days


# TODO: quantities

def main(argv):
    json_file_path = ""

    is_setup = False
    is_partial = False
    is_recombine = False
    part = 0
    try:
        (opts, args) = getopt.getopt(argv, "hf:sp:r")
    except getopt.GetoptError, e:
        print "Wrong options"
        print e
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            json_file_path = arg
        if opt in ("-h", "--help"):
            sys.exit()
        if opt in ("-s", "--setup"):
            is_setup = True
        if opt in ("-p", "--part"):
            is_partial = True
            part = int(arg)
        if opt in ("-r", "--recombine"):
            is_recombine = True

    if json_file_path == "":
        print "No file."
        sys.exit()

    config = ConfigurationFile(json_file_path)

    # places it in the base_prefix data subdir.
    prefix = '%s_c%d_p%d_d%d_int%d_pmod_%s_' % (config.base_prefix,
                                                config.clients_count,
                                                config.products_count,
                                                config.days_count,
                                                config.day_interval,
                                                config.model_name)
    print prefix

    if is_setup:
        # doesn't generate the orders
        clients, products, _, model = generate_dataset(clients_count=config.clients_count,
                                                       products_count=config.products_count,
                                                       days_count=0,
                                                       day_interval=0,
                                                       model_name=config.model_name)
        save_all(clients, products, {}, model, config.base_prefix, prefix)
    elif is_partial:
        # load the data, select only the right part and generates the orders
        clients, products, _, model = load_all(config.base_prefix, prefix)

        part_from = config.part_size * part
        part_to = config.part_size * (part+1)

        # restrict the clients set and reset the index so it goes from 0 to len-1
        # the old index is save in a new column 'index'
        clients = clients.iloc[part_from: part_to, :].reset_index()

        days = generate_days(config.days_count, config.day_interval)

        orders = generate_orders(clients, products, days, model)
        prefix = 'part%d_%s' % (part, prefix)
        save_all(clients, products, orders, model, config.base_prefix, prefix)
    elif is_recombine:
        # recombines the partial orders in a single file
        # loads the clients and the products
        clients, products, _, model = load_all(config.base_prefix, prefix)
        total_parts = config.clients_count / config.part_size
        partial_prefixes = ['part%d_%s' % (i, prefix) for i in range(0, total_parts)]
        # merge the partials and generates the complete file
        save_partial_orders(config.base_prefix, partial_prefixes, prefix)
    else:
        clients, products, orders, model = generate_dataset(clients_count=config.clients_count,
                                                            products_count=config.products_count,
                                                            days_count=config.days_count,
                                                            day_interval=config.day_interval,
                                                            model_name=config.model_name)
        save_all(clients, products, orders, model, config.base_prefix, prefix)


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
