# -*- coding: utf-8 -*-
import random

import datetime

import math
import numpy as np
from sklearn.datasets import make_classification


from dataset.generator.order import Order
from dataset.generator.polynomial import PeriodicPolynomial, PeriodicThresholdFunction, PeriodicFunction
from dataset.generator.product import Product
from dataset.generator.values import SECS_IN_DAY, CLIENT_KIND_REGULAR, CLIENT_KIND_CONSUMPTION, CLIENT_TYPE_GDO, \
    CLIENT_TYPE_RESTURANT, PERIOD_LENGTH
from util import weighted_choice


class ClientFactory(object):
    @staticmethod
    def create_client(id, products, client_kind, client_type, high_consumption_ratio, no_consumption_ratio,
                      order_days=None):
        if client_kind == CLIENT_KIND_REGULAR:
            return ClientFactory.create_regular_client(id,
                                                       products,
                                                       client_type,
                                                       order_days,
                                                       high_consumption_ratio,
                                                       no_consumption_ratio)
        elif client_kind == CLIENT_KIND_CONSUMPTION:
            return ClientFactory.create_consumption_client(id,
                                                           products,
                                                           client_type,
                                                           high_consumption_ratio,
                                                           no_consumption_ratio)
        else:
            # client_kind == RANDOM
            return ClientFactory.create_random_client(id,
                                                      products,
                                                      client_type,
                                                      high_consumption_ratio,
                                                      no_consumption_ratio)

    @staticmethod
    def create_regular_client(id, products, client_type, order_days, high_consumption_ratio, no_consumption_ratio):
        return RegularClient(id=id,
                             products=products,
                             client_type=client_type,
                             order_days=order_days,
                             high_consumption_ratio=high_consumption_ratio,
                             no_consumption_ratio=no_consumption_ratio)

    @staticmethod
    def create_consumption_client(id, products, client_type, high_consumption_ratio, no_consumption_ratio):
        return ConsumptionClient(id=id,
                                 products=products,
                                 client_type=client_type,
                                 high_consumption_ratio=high_consumption_ratio,
                                 no_consumption_ratio=no_consumption_ratio)

    @staticmethod
    def create_random_client(id, products, client_type, high_consumption_ratio, no_consumption_ratio):
        return RandomClient(id=id,
                            products=products,
                            client_type=client_type,
                            high_consumption_ratio=high_consumption_ratio,
                            no_consumption_ratio=no_consumption_ratio)

class Client(object):
    """
    Classe che simula il comportamento di un cliente
    """

    # Parte statica

    SYNTETIC_FEATURE_COUNT = 10

    @staticmethod
    def get_dataframe_header():
        header = ['client_name', 'client_id']
        for f in range(0,Client.SYNTETIC_FEATURE_COUNT):
            header += ['client_feature_'+str(f)]
        return header

    # ---- Fine parte statica
    #

    def __setup_period_consumption(self):
        """
        Metodo che configura i consumi periodici (settimanali) in base alla tipologia di cliente
        :return: 
        """
        if self.client_type == CLIENT_TYPE_GDO:
            self.period_consumption = PeriodicPolynomial(degree=10, theta=1, build_mode='fit', points=4)
        elif self.client_type == CLIENT_TYPE_RESTURANT:
            points = [
                (0, random.uniform(0.5, 1.2)),
                (1, random.uniform(0.5, 1.2)),
                (2, random.uniform(0.5, 1.2)),
                (3, random.uniform(0.5, 1.2)),
                (4, random.uniform(1, 2)),  # Consumi più alti nel weekend
                (5, random.uniform(1, 3)),
                (6, random.uniform(1, 2))
            ]
            # giorno di riposo infra-settimanale, conusmi a 0
            rest_day = random.randint(0, 3)
            points[rest_day] = (rest_day, 0)
            self.period_consumption = PeriodicThresholdFunction.create_from_points(points)
        else:
            # type == CLIENT_TYPE_SHOP
            points = [(i, random.uniform(0.5, 1.2)) for i in range(0, 7)]
            rest_day = random.randint(0, 6)
            points[rest_day] = (rest_day, 0)
            self.period_consumption = PeriodicThresholdFunction.create_from_points(points)

    def __setup_product_consumption(self):
        high_consumption_count = int(math.floor(self.high_consumption_ratio * self.prod_count))
        no_consumption_count = int(math.floor(self.no_consumption_ratio * self.prod_count))

        #print high_consumption_count, no_consumption_count

        product_ids = [i for i in range(0, self.prod_count)]
        # scelgo a caso gli elementi che non vado a consumare
        for i in range(0, no_consumption_count):
            p = random.choice(product_ids)
            product_ids.remove(p)
            self.consumptions[p] = 0
        assert len(product_ids) == len(self.products) - no_consumption_count
        weighted_ids = [(p, self.products[p].popularity_index) for p in product_ids]
        for i in range(0, high_consumption_count):
            p = weighted_choice(weighted_ids)
            weighted_ids.remove((p, self.products[p].popularity_index))
            product_ids.remove(p)
            self.consumptions[p] = random.uniform(0.25, 0.5)

        assert len(product_ids) == len(self.products) - no_consumption_count - high_consumption_count

        for p in product_ids:
            self.consumptions[p] = random.uniform(0.01, 0.15)

    def __setup_syntetic_fields(self):
        X, _ = make_classification(n_samples=1,
                                   n_features=Client.SYNTETIC_FEATURE_COUNT,
                                   n_informative=4,
                                   n_repeated=2,
                                   n_classes=4)
        self.extra_feature = X[0]  # ndarray di una sola riga

    def __init__(self, id, products, client_type, high_consumption_ratio, no_consumption_ratio):
        # type: (int, [Product], str) -> None
        self.id = id
        self.name = 'client_' + str(id)
        self.prod_count = len(products)
        self.products = products
        self.storage = np.ones(shape=(self.prod_count,))

        # Impostazione dei consumi dei vari prodotti
        assert no_consumption_ratio + high_consumption_ratio <= 1
        self.no_consumption_ratio = no_consumption_ratio
        self.high_consumption_ratio = high_consumption_ratio
        self.consumptions = np.ones(shape=(self.prod_count,))
        self.__setup_product_consumption()

        # Impostazione del consumo periodico in base al tipo di cliente
        self.client_type = client_type
        self.__setup_period_consumption()

        # Impostazione dei campi sintetici
        self.__setup_syntetic_fields()

    def to_dict(self):
        # type: () -> dict
        """
        Converte il cliente in un dizionario che può essere utilizzato per inserire
        il cliente in un dataframe per l'apprendimento.
        
        NOTA: vengono perse tutte le informazioni utili per generare gli ordini (polinomi,
        consumi, magazzino, ecc.).
        :return: dizionario rappresentante il cliente
        """
        client_dict = {
            'client_id': self.id,
            'client_name': self.name
        }
        for f in range(0,Client.SYNTETIC_FEATURE_COUNT):
            client_dict['client_feature_'+str(f)] = self.extra_feature[f]
        return client_dict

    def generate_orders(self, from_ts, to_ts, global_trend):
        # type: (long, long, PeriodicPolynomial) -> [Order]
        pass


class ConsumptionClient(Client):
    def __init__(self, id, products, client_type, high_consumption_ratio, no_consumption_ratio):
        super(ConsumptionClient, self).__init__(id, products, client_type, high_consumption_ratio, no_consumption_ratio)

    def generate_orders(self, from_ts, to_ts, global_trend):
        # type: (long, long, PeriodicPolynomial) -> [Order]
        orders = []
        for ts in range(from_ts, to_ts + SECS_IN_DAY, SECS_IN_DAY):
            #print "--- Day ", ts, "Storage", self.storage
            for p in range(0, self.prod_count):
                scaled_t = float(datetime.datetime
                                 .fromtimestamp(ts)
                                 .timetuple().tm_yday  # estra il day-of-year del t corrente
                                 ) / PERIOD_LENGTH  # scala il DOY nel range 0..1
                # Controllo che il prodotto sia in stagione, se sono fuori stagione non lo ordino
                # ovvero assumo che il cliente sia sufficientemente sveglio da conoscere la stagionalità
                # dei prodotti che ordina
                if self.products[p].market_availability(scaled_t) == 0:
                    continue
                prod_consumption = self.period_consumption(scaled_t) \
                                   * self.consumptions[p] \
                                   * self.products[p].perishability_scale \
                                   * global_trend(scaled_t) \
                                   * random.normalvariate(1, 0.2)
                self.storage[p] -= prod_consumption
                if self.storage[p] <= 0:
                    # fa l'ordine

                    #########
                    # Assumo che l'ordine vada sempre a buon fine
                    requested_qty = 1
                    received_qty = 1
                    #########
                    # Assumo di ordinare tutta la merce che mi serve per riempire il magazzino
                    # e che l'ordine non mi garantisca il riempimento del magazzino.
                    # requested_qty = 1 - self.storage[p]
                    # received_qty = self.products[p].available_quantity(t)
                    #######
                    self.storage[p] = received_qty

                    orders.append(Order(timestamp=ts,
                                        client_id=self.id,
                                        product_id=p,
                                        requested_qty=requested_qty,
                                        received_qty=received_qty))
                    #print "Ordered product_%d qty %f - received %f" % (p, requested_qty, received_qty)

        return orders


class RegularClient(Client):
    def __init__(self, id, products, client_type, high_consumption_ratio, no_consumption_ratio, order_days):
        # order_days -> lista di interi che rappresenta quando si può fare un ordine
        super(RegularClient, self).__init__(id, products, client_type, high_consumption_ratio, no_consumption_ratio)
        self.order_days = sorted(order_days)

    def generate_orders(self, from_ts, to_ts, global_trend):
        # type: (long, long, PeriodicPolynomial) -> [Order]

        orders = []

        # Non posso usare questo approccio, perché il timestamp 0 è a l'una di mattina ora locale e ciò sfasa i conti
        # from_day = int(math.floor(from_ts / SECS_IN_DAY))
        # to_day = int(math.floor(to_ts / SECS_IN_DAY))
        # for t in range(from_day, to_day + 1): # Non posso usare

        for ts in range(from_ts, to_ts + SECS_IN_DAY, SECS_IN_DAY):
            #print "--- Day ", ts, "Storage", self.storage
            #ts = t * SECS_IN_DAY
            dow = datetime.datetime.fromtimestamp(ts).timetuple().tm_wday
            if dow in self.order_days:
                # è un giorno buono per effettuare gli ordini

                # determino tra quanto sarà il prossimo giorno buono per ordinare
                next_order_day_index = self.order_days.index(dow) + 1
                if next_order_day_index < len(self.order_days):
                    next_order_day = self.order_days[next_order_day_index]
                else:
                    next_order_day = self.order_days[0]
                if next_order_day > dow:
                    days = next_order_day - dow
                else:
                    days = next_order_day + 7 - dow

                for p in range(0, self.prod_count):
                    scaled_t = float(datetime.datetime
                                     .fromtimestamp(ts)
                                     .timetuple().tm_yday  # estra il day-of-year del t corrente
                                     ) / PERIOD_LENGTH  # scala il DOY nel range 0..1
                    # Controllo che il prodotto sia in stagione, se sono fuori stagione non lo ordino
                    # ovvero assumo che il cliente sia sufficientemente sveglio da conoscere la stagionalità
                    # dei prodotti che ordina
                    if self.products[p].market_availability(scaled_t) == 0:
                        continue

                    prod_consumption = 0
                    for d in range(ts,
                                   ts + (days + 1) * SECS_IN_DAY,
                                   SECS_IN_DAY):  # considero il consumo per i prossimi giorni, fino al prossimo giorno buono per ordinare
                        scaled_d = float(d) / PERIOD_LENGTH
                        prod_consumption += self.period_consumption(scaled_d) \
                                            * self.consumptions[p] \
                                            * self.products[p].perishability_scale \
                                            * global_trend(scaled_d) \
                                            * random.normalvariate(1, 0.2)
                    self.storage[p] -= prod_consumption
                    if self.storage[p] <= 0:
                        # fa l'ordine

                        #########
                        # Assumo che l'ordine vada sempre a buon fine
                        requested_qty = 1
                        received_qty = 1
                        #########
                        # Assumo di ordinare tutta la merce che mi serve per riempire il magazzino
                        # e che l'ordine non mi garantisca il riempimento del magazzino.
                        # requested_qty = 1 - self.storage[p]
                        # received_qty = self.products[p].available_quantity(t)
                        #######
                        self.storage[p] = received_qty

                        orders.append(Order(timestamp=ts,
                                            client_id=self.id,
                                            product_id=p,
                                            requested_qty=requested_qty,
                                            received_qty=received_qty))
                        #print "Ordered product_%d qty %f - received %f" % (p, requested_qty, received_qty)
        return orders


class RandomClient(Client):
    def __init__(self, id, products, client_type, high_consumption_ratio, no_consumption_ratio):
        super(RandomClient, self).__init__(id, products, client_type, high_consumption_ratio, no_consumption_ratio)

    def generate_orders(self, from_ts, to_ts, global_trend):
        # type: (long, long, PeriodicPolynomial) -> [Order]

        orders = []

        for ts in range(from_ts, to_ts + SECS_IN_DAY, SECS_IN_DAY):
            for p in range(0, self.prod_count):
                scaled_t = float(datetime.datetime
                                 .fromtimestamp(ts)
                                 .timetuple().tm_yday  # estra il day-of-year del t corrente
                                 ) / PERIOD_LENGTH  # scala il DOY nel range 0..1
                # Controllo che il prodotto sia in stagione, se sono fuori stagione non lo ordino
                # ovvero assumo che il cliente sia sufficientemente sveglio da conoscere la stagionalità
                # dei prodotti che ordina
                if self.products[p].market_availability(scaled_t) == 0:
                    continue

                order_probability = global_trend(scaled_t) * self.consumptions[p]
                if order_probability > 1:
                    order_probability = 1

                if random.random() <= order_probability:
                    # Faccio l'ordine
                    orders.append(Order(timestamp=ts,
                                        client_id=self.id,
                                        product_id=p,
                                        requested_qty=1,
                                        received_qty=1))
                    #print "Ordered product_%d qty %f - received %f" % (p, 1, 1)

        return orders
