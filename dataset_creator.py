#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getopt
import sys
import time
import datetime
import pandas as pd

from generator.probability_models import ProbabilityModel
from util.file_helper import ConfigurationFile, save_train_set, save_test_set, load_train_set, \
    merge_partial_orders, D_TRAIN, D_VERSION, D_TEST
from generator.generators import generate_dataset, generate_orders, generate_days

NUMBER_OF_TEST_SET_VERSION = 5

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
        # Nella modalità `setup` non vengono generati gli ordini
        clients, products, _, model = generate_dataset(clients_count=config.clients_count,
                                                       products_count=config.products_count,
                                                       days_count=0,  # non genera gli ordini
                                                       day_interval=0,
                                                       model_name=config.model_name)
        save_train_set(clients, products, {}, model, config.base_prefix, prefix)
    elif is_partial:
        # Nella modalità `partial` vengono caricati i dati (tranne gli ordini) e vengono generati
        # gli ordini relativi solamente alla parte di clienti specificata.
        clients, products, _, model = load_train_set(config.base_prefix, prefix)

        part_from = config.part_size * part
        part_to = config.part_size * (part+1)
        # Restringe il dataset dei clienti e reimposta l'indice in modo che vada da 0 a len-1
        # il vecchio indice viene mantenuto nella nuova colonna `index`
        clients = clients.iloc[part_from: part_to, :].reset_index()

        days = generate_days(config.days_count, config.day_interval)

        orders = generate_orders(clients, products, days, model)
        prefix = 'part%d_%s' % (part, prefix)

        save_train_set(clients, products, orders, model, config.base_prefix, prefix)
        generate_test_set(clients, products, model, config, prefix)
    elif is_recombine:
        # Nella modalità `recombine` vengono uniti i vari dataset parziali in un unico file

        total_parts = config.clients_count / config.part_size
        partial_prefixes = ['part%d_%s' % (i, prefix) for i in range(0, total_parts)]

        # merge the partials and generates the complete file
        merge_partial_orders(config.base_prefix+'/'+D_TRAIN, partial_prefixes, prefix)

        for i in range(0, NUMBER_OF_TEST_SET_VERSION):
            base_prefix = config.base_prefix+'/'+D_TEST+'/'+D_VERSION+str(i)
            merge_partial_orders(base_prefix, partial_prefixes, prefix)
    else:
        # Nessun caricamento, il dataset viene generato live.
        clients, products, orders, model = generate_dataset(clients_count=config.clients_count,
                                                            products_count=config.products_count,
                                                            days_count=config.days_count,
                                                            day_interval=config.day_interval,
                                                            model_name=config.model_name)
        save_train_set(clients, products, orders, model, config.base_prefix, prefix)
        generate_test_set(clients, products, model, config, prefix)


def generate_test_set(clients, products, model, config, prefix):
    # type: (pd.DataFrame, pd.DataFrame, ProbabilityModel, ConfigurationFile, str) -> None
    """
    Genera 5 possibili test set per i dati. Ogni dataset è composto da 2 matrici, una relativa
     agli ordini per oggi e una relativi a quelli per dopo domani.
    """
    print "Generating test set..."
    today_timestamp = time.time()  # current timestamp
    after_tomorrow = datetime.date.today() + datetime.timedelta(days=2)
    after_tomorrow_timestamp = time.mktime(after_tomorrow.timetuple())
    days = [today_timestamp, after_tomorrow_timestamp]

    for i in range(0, NUMBER_OF_TEST_SET_VERSION):
        print "Version", i
        orders = generate_orders(clients, products, days, model)
        save_test_set(clients, products, orders, config.base_prefix, prefix, version=i)
    return


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
