#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.file_helper import save_all, load_all
from generator.generators import generate_dataset


# TODO: quantities

#################################################################################################

clients_count = 10
products_count = 100
days_count = 10
day_interval = 0  # continuous
model_name = 'cond'  # 'random'

prefix = 'c%d_p%d_d%d_int%d_pmod_%s_' % (clients_count, products_count, days_count, day_interval, model_name)
print prefix


clients, products, orders, model = generate_dataset(clients_count=clients_count,
                                                    products_count=products_count,
                                                    days_count=days_count,
                                                    day_interval=day_interval,
                                                    model_name=model_name)
save_all(clients, products, orders, model, prefix)


#################################################################################################

clients_count = 10
products_count = 100
days_count = 100
day_interval = 4
model_name = 'cond'  # 'random'

prefix = 'c%d_p%d_d%d_int%d_pmod_%s_' % (clients_count, products_count, days_count, day_interval, model_name)
print prefix

clients, products, orders, model = generate_dataset(clients_count=clients_count,
                                                    products_count=products_count,
                                                    days_count=days_count,
                                                    day_interval=day_interval,
                                                    model_name=model_name)
save_all(clients, products, orders, model, prefix)


#################################################################################################

clients_count = 1000
products_count = 1000
days_count = 365
day_interval = 0  # continuous
model_name = 'cond'  # 'random'

prefix = 'c%d_p%d_d%d_int%d_pmod_%s_' % (clients_count, products_count, days_count, day_interval, model_name)
print prefix

clients, products, orders, model = generate_dataset(clients_count=clients_count,
                                                    products_count=products_count,
                                                    days_count=days_count,
                                                    day_interval=day_interval,
                                                    model_name=model_name)
save_all(clients, products, orders, model, prefix)


#################################################################################################

clients_count = 1000
products_count = 1000
days_count = 365
day_interval = 4  # continuous
model_name = 'cond'  # 'random'

prefix = 'c%d_p%d_d%d_int%d_pmod_%s_' % (clients_count, products_count, days_count, day_interval, model_name)
print prefix

clients, products, orders, model = generate_dataset(clients_count=clients_count,
                                                    products_count=products_count,
                                                    days_count=days_count,
                                                    day_interval=day_interval,
                                                    model_name=model_name)
save_all(clients, products, orders, model, prefix)
