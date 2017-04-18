import random


def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    for c, w in choices:
        r -= w
        if r <= 0:
            return c
    assert False, "Errore nella distribuzione"
