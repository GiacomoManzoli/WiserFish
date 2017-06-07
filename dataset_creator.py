#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getopt
import sys
import time

from dataset.generator.generator_v2 import generate_dataset, generate_orders
from dataset.manager import save_dataset, save_generators, load_generators, save_partial_orders, \
    merge_partial_orders_and_save, save_configuration_file
from dataset.generator.values import SECS_IN_DAY

from util.file_helper import ConfigurationFile


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

    if is_setup:
        # Nella modalità `setup` non vengono generati gli ordini
        clients, products, _, global_trend = \
            generate_dataset(product_count=config.products_count,
                             seasonal_ratio=config.seasonal_ratio,
                             client_count=config.clients_count,
                             regular_ratio=config.regular_ratio,
                             consumption_ratio=config.consumption_ratio,
                             gdo_ratio=config.gdo_ratio,
                             resturant_ratio=config.resturant_ratio,
                             from_ts=config.starting_day,
                             to_ts=config.starting_day) # 0 giorni

        save_generators(clients, products, global_trend, dataset_name=config.base_prefix)
        save_configuration_file(config)
    elif is_partial:
        # Nella modalità `partial` vengono caricati i dati dei generatori e vengono generati
        # gli ordini per un determianto intervallo di tempo
        print "Generazione degli ordini parte", part
        clients, _, global_trend = load_generators(config.base_prefix)

        from_ts = config.starting_day
        to_ts = config.starting_day + (config.days_count * SECS_IN_DAY)

        part_from = config.part_size * part
        part_to = config.part_size * (part + 1)
        # Restringe i clienti
        clients = clients[part_from:part_to]

        orders = generate_orders(clients, from_ts, to_ts, global_trend)

        ts = {}
        for o in orders:
            if o.timestamp not in ts.keys():
                ts[o.timestamp] = 1
        # Controllo che ci siano ordini solo per l'intervallo giusto

        save_partial_orders(dataset_name=config.base_prefix, orders=orders, part_index=part)

    elif is_recombine:
        # Nella modalità `recombine` vengono uniti i vari dataset parziali in un unico file

        total_parts = config.clients_count / config.part_size
        merge_partial_orders_and_save(dataset_name=config.base_prefix,
                                      parts_count=total_parts)
    else:
        # Nessun caricamento, il dataset viene generato live.
        print "Genero il dataset..."
        clients, products, orders, global_trend = \
            generate_dataset(product_count=config.products_count,
                             seasonal_ratio=config.seasonal_ratio,
                             client_count=config.clients_count,
                             regular_ratio=config.regular_ratio,
                             consumption_ratio=config.consumption_ratio,
                             gdo_ratio=config.gdo_ratio,
                             resturant_ratio=config.resturant_ratio,
                             from_ts=config.starting_day,
                             to_ts=config.starting_day+(config.days_count * SECS_IN_DAY))
        print "Dataset generato. Salvo su file."
        save_dataset(clients, products, orders, global_trend, dataset_name=config.base_prefix)
        save_configuration_file(config)


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
