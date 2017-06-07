# -*- coding: utf-8 -*-
import random


def weighted_choice(choices):
    # type: ([(object, float)]) -> object
    """
    Effettua la scelta pesata tra le possibili scelte.
    :param choices: coppie (v,p) dove p è la probabilità del valore v
    :return: 
    """
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    for c, w in choices:
        r -= w
        if r <= 0:
            return c
    assert False, "Errore nella distribuzione"
