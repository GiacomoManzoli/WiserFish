#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getopt
import sys
import time

import pandas as pd
import sqlite3

from datetime import datetime, date

from dataset.generator.values import SECS_IN_DAY
from util import ConfigurationFile


def main(argv):
    load_from_file = False
    json_file_path = ""

    try:
        (opts, args) = getopt.getopt(argv, "f:")
    except getopt.GetoptError, e:
        print "Wrong options"
        print e
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            load_from_file = True
            json_file_path = arg

    if not load_from_file:
        print "Errore: non è stato specificato il file!"
        return

    config = ConfigurationFile(json_file_path)


    load_start_time = time.time()

    # Dataframe contenti i clienti, prodotti e ordini
    clients_df = pd.read_csv('./datasets/%s/data/clients.csv' % (config.base_prefix))
    products_df = pd.read_csv('./datasets/%s/data/products.csv' % (config.base_prefix))
    orders_df = pd.read_csv('./datasets/%s/data/orders.csv' % (config.base_prefix))

    print "Dataset loaded!"
    print "Total duration: ", time.time() - load_start_time, "seconds"

    dataset_star_ts = config.starting_day
    dataset_end_ts = long(dataset_star_ts + (config.days_count - 1) * SECS_IN_DAY)

    print orders_df.head(5)

    cnx = sqlite3.connect('./datasets/%s/data/%s.db' % (config.base_prefix, config.base_prefix))

    products_df = products_df.set_index(['product_id'])
    clients_df = clients_df.set_index(['client_id'])

    orders_df['order_date'] = orders_df.apply(lambda row: datetime.fromtimestamp(row['datetime']), axis=1)
    orders_df['year'] = orders_df.apply(lambda row: row['order_date'].year, axis=1)

    orders_df['day_of_year'] = orders_df.apply(lambda row: row['order_date'].toordinal()
                                                 - date(row['order_date'].year, 1, 1).toordinal() + 1, axis=1)
    orders_df = orders_df.drop('order_date', axis=1)

    orders_types = {
        'order_id':'INTEGER',
        'datetime':'INTEGER',
        'client_id':'INTEGER',
        'product_id':'INTEGER',
        'requested_qty':'REAL',
        'received_qty':'REAL',
        'year':'INTEGER',
        'day_of_year':'INTEGER'
    }

    orders_df.to_sql(name='orders',
                     con=cnx,
                     chunksize=40,
                     if_exists='replace',
                     index=True,
                     index_label='order_id',
                     dtype=orders_types)
    products_df.to_sql(name='products', con=cnx, chunksize=40, if_exists='replace', index=True, index_label='product_id')
    clients_df.to_sql(name='clients', con=cnx, chunksize=40, if_exists='replace', index=True, index_label='client_id')


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"
