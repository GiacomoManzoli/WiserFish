# -*- coding: utf-8 -*-
import random

from sklearn.datasets import make_classification

from dataset.generator.polynomial import PeriodicPolynomial, PeriodicThresholdFunction
from dataset.generator.values import PERIOD_LENGTH


class Product(object):

    # Parte statica

    SYNTETIC_FEATURE_COUNT = 15

    @staticmethod
    def get_dataframe_header():
        header = ['product_name', 'product_id']
        for f in range(0,Product.SYNTETIC_FEATURE_COUNT):
            header += ['product_feature_'+str(f)]
        return header

    # ---- Fine parte statica
    #

    def __setup_syntetic_fields(self):
        features = 15
        X, _ = make_classification(n_samples=1,
                                   n_features=Product.SYNTETIC_FEATURE_COUNT,
                                   n_informative=4,
                                   n_repeated=2,
                                   n_classes=4)
        self.extra_feature = X[0]  # ndarray di una sola riga

    def __init__(self, id, seasonal_product=False):
        self.id = id
        self.name = 'product_' + str(id)
        self.seasonal_product = seasonal_product

        # Quantità disponibile del prodotto
        self.availability = PeriodicPolynomial(degree=30, theta=1, build_mode='fit', points=5)
        # Disponibilità nel mercato (stagionalità del prodotto)
        if self.seasonal_product:
            # Definisco un periodo casuale in cui il prodotto è disponibile
            season_start = random.uniform(0, 1)
            season_end = random.uniform(0, 1)
            # Mi assicuro che l'intervallo abbia senso
            if season_start == season_end:
                season_end = season_start + 0.1 if season_start + 0.1 <= 1 else 1
            elif season_end < season_start:
                temp = season_start
                season_start = season_end
                season_end = temp

            self.market_availability = PeriodicThresholdFunction(
                points=[(0, 0), (season_start, 1), (season_end, 0), (1, 0)])

        else:
            # Prodotto sempre disponibile
            self.market_availability = PeriodicThresholdFunction(points=[(0, 1), (1, 1)])

        self.perishability_scale = random.normalvariate(1, 0.25)
        # Indice di deperibilità del prodotto, usato per scalare i consumi dei vari clienti
        # L'idea è che se un prodotto è deperibile, il cliente cerca di consumarlo più in fretta

        self.popularity_index = random.uniform(0, 1)
        # Indice di popolarità del prodotto, viene utilizzato per determinare quali prodotti
        # sono più consumati dai clienti

        # Aggiunge delle feature sintentiche
        self.__setup_syntetic_fields()

    def __str__(self):

        return "Product: <Name: %s Seasonal: %d Perishability: %f Popularity: %f>" % \
               (self.name, self.seasonal_product, self.perishability_scale, self.popularity_index)

    def available_quantity(self, t):
        scaled_t = float(t) / PERIOD_LENGTH
        return self.availability(scaled_t) * self.market_availability(scaled_t)

    def to_dict(self):
        # type: () -> dict
        """
        Converte il prodotto in un dizionario che può essere utilizzato per inserire
        il prodotto in un dataframe per l'apprendimento.

        NOTA: vengono perse tutte le informazioni utili per generare gli ordini.
        :return: dizionario rappresentante il prodotto
        """
        product_dict = {
            'product_id': self.id,
            'product_name': self.name
        }
        for f in range(0, Product.SYNTETIC_FEATURE_COUNT):
            product_dict['product_feature_' + str(f)] = self.extra_feature[f]
        return product_dict