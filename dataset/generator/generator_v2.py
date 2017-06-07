# -*- coding: utf-8 -*-
import math
import random

from dataset.generator.client import ClientFactory, Client
from dataset.generator.order import Order
from dataset.generator.polynomial import PeriodicPolynomial
from dataset.generator.product import Product
from values import *


def generate_dataset(product_count, seasonal_ratio,
                     client_count, regular_ratio, consumption_ratio, gdo_ratio, resturant_ratio,
                     from_ts, to_ts):
    # type: (int, float, int, float, float, float, float, long, long) -> ([Client], [Product], [Order], PeriodicPolynomial)

    # Genera un trend per gli ordini
    global_trend = PeriodicPolynomial(degree=20, theta=1, build_mode='fit', points=20)

    # Genera i prodotti
    products = __generate_products(product_count=product_count,
                                   seasonal_ratio=seasonal_ratio)
    # Genera i clienti
    clients = __generate_clients(client_count=client_count,
                                 regular_ratio=regular_ratio,
                                 consumption_ratio=consumption_ratio,
                                 gdo_ratio=gdo_ratio,
                                 resturant_ratio=resturant_ratio,
                                 products=products)
    # Genera gli ordini
    orders = generate_orders(clients, from_ts, to_ts, global_trend)

    return clients, products, orders, global_trend


def generate_orders(clients, from_ts, to_ts, global_trend):
    orders = []
    for client in clients:
        print "Ordini per il cliente ", client.name
        orders += client.generate_orders(from_ts, to_ts, global_trend)
    return orders


####
# Metodi ausiliari per la generazione del dataset
####

def __generate_products(product_count, seasonal_ratio=0.75):
    print "Genero i prodotti..."
    products = []

    seasonals = [True] * int(math.floor(product_count * seasonal_ratio))
    seasonals += [False] * (product_count - int(math.floor(product_count * seasonal_ratio)))
    random.shuffle(seasonals)  # shuffle in-place

    assert len(seasonals) == product_count

    for p in range(0, product_count):
        products.append(Product(id=p,
                                seasonal_product=seasonals[p]))

    assert len(products) == product_count

    return products


def __generate_clients(client_count, regular_ratio, consumption_ratio, gdo_ratio, resturant_ratio, products):
    """
    Genera `client_count` clienti, di cui `regular_ratio` si effettuano ordini in modo regolare e `consumption_ratio`
    effettuano ordini quando finiscono il magazzino. Se `regular_ratio` + `consumption_ratio` != 1, i restanti clienti
    si comportano in modo casuale.
    `gdo_ratio` e `resturant_ratio` si comportano in modo analogo, con la differenza che se non sommano a 1, i restanti
    clienti vengono creato come negozi (CLIENT_TYPE_SHOP).
    Il CLIENT_KIND e la modalità d'ordine sono scorrelati, ovvero un cliente GDO può effettuare ordini a consumo o a 
    giorni fissi. 
    """
    # TODO-AskAndrea: ha senso forzare una correlazione?
    """
    
    Nota: nel costruire un cliente servono anche: `high_consumption_ratio`, `no_consumption_ratio` e `orders_day`.
    Questi tre valori vengono scelti casualemente all'interno di determinati intervalli.
    
    :param client_count: numero di clienti da generare
    :param regular_ratio: rapporto di clienti con ordini regolari
    :param consumption_ratio: rapporto di clienti con ordini a necessità
    :param gdo_ratio: rapporto di clienti che si comportano come la GDO
    :param resturant_ratio: rapporto di clienti che si comportano come un ristorante
    :return: clienti generati
    """
    print "Genero i clienti..."

    clients = []

    # Tipo di clienti: regolari, random, a consumo
    regular_count = int(math.floor(client_count * regular_ratio))
    consumption_count = int(math.floor(client_count * consumption_ratio))
    random_count = client_count - regular_count - consumption_count
    assert random_count >= 0
    kinds = [CLIENT_KIND_REGULAR] * regular_count
    kinds += [CLIENT_KIND_CONSUMPTION] * consumption_count
    kinds += [CLIENT_KIND_RANDOM] * random_count
    random.shuffle(kinds)
    # Tipo di clienti: GDO, ristoranti, negozi
    gdo_count = int(math.floor(client_count * gdo_ratio))
    resturant_count = int(math.floor(client_count * resturant_ratio))
    shop_count = client_count - gdo_count - resturant_count
    assert shop_count >= 0
    types = [CLIENT_TYPE_GDO] * gdo_count
    types += [CLIENT_TYPE_RESTURANT] * resturant_count
    types += [CLIENT_TYPE_SHOP] * shop_count
    random.shuffle(types)

    for c in range(0, client_count):

        # conumption ratio
        # TODO-AskAndrea: può andare come stima?
        hcr = random.uniform(0.01, 0.10)
        ncr = random.uniform(0.5, 1 - hcr)

        order_days = None
        if (kinds[c] == CLIENT_KIND_REGULAR):
            # determino dei giorni in cui può ordinare
            good_days = random.randint(1, len(ALL_DAYS))  # stabilisco quante volte alla settimana il cliente ordina
            days = list(ALL_DAYS)  # prendo tutti i giorni
            random.shuffle(days)  # rimescolo i giorni, così li scelgo casualemente
            order_days = days[0:good_days]  # prendo i primi `good_days` giorni come giorni buoni per ordinare

        clients.append(ClientFactory.create_client(id=c,
                                                   products=products,
                                                   high_consumption_ratio=hcr,
                                                   no_consumption_ratio=ncr,
                                                   client_kind=kinds[c],
                                                   client_type=types[c],
                                                   order_days=order_days))

    return clients
