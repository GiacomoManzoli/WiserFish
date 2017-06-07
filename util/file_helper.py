# -*- coding: utf-8 -*-

import json
import datetime
import time

################################################################
# CONFIGURATION FILE HELPER
################################################################

# JSON field names
J_PREFIX = "prefix"
J_CLIENTS_COUNT = "clients_count"
J_CLIENT_RESTURANT_RATIO="client_resturant_ratio"
J_CLIENT_GDO_RATIO="client_gdo_ratio"
J_CLIENT_REGULAR_RATIO="client_regular_ratio"
J_CLIENT_CONSUNPTION_RATIO="client_consumption_ratio"

J_PRODUCTS_COUNT = "products_count"
J_PRODUCT_SEASONAL_RATIO="product_seasonal_ratio"

J_DAYS_COUNT = "days_count"
J_STARTING_DAY = "starting_day"

J_PART_SIZE = "part_size"


class ConfigurationFile(object):
    def __init__(self, json_file_path):

        self.file_path = json_file_path

        print "Loading configuration from: ", json_file_path
        with open(json_file_path) as json_file:
            json_data = json.load(json_file)
        print json_data

        self.base_prefix = json_data[J_PREFIX]

        self.clients_count = json_data[J_CLIENTS_COUNT]
        self.resturant_ratio = json_data[J_CLIENT_RESTURANT_RATIO]
        self.gdo_ratio = json_data[J_CLIENT_GDO_RATIO]
        self.regular_ratio = json_data[J_CLIENT_REGULAR_RATIO]
        self.consumption_ratio = json_data[J_CLIENT_CONSUNPTION_RATIO]

        self.products_count = json_data[J_PRODUCTS_COUNT]
        self.seasonal_ratio = json_data[J_PRODUCT_SEASONAL_RATIO]

        self.days_count = json_data[J_DAYS_COUNT]

        str_starting_time = json_data[J_STARTING_DAY]
        self.starting_day = long(time.mktime(
            datetime.datetime.strptime(str_starting_time, "%Y-%m-%d").timetuple()
        ))

        self.part_size = json_data[J_PART_SIZE]
