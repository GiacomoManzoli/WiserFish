# -*- coding: utf-8 -*-
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P


class Polynomial(object):
    """
    Classe che rappresenta un polinomio generico
    """

    def __init__(self, degree):
        # type: (int) -> None
        """
        Costruisce un polinomio di grado `degree` con coefficienti casuali 
        :param grade: 
        """
        self.degree = degree
        self.a = []
        for i in range(0, degree + 1):
            self.a +=[random.uniform(-1, 1)]

    def __call__(self, x):
        return self.calculate(x)

    def __str__(self):
        res = ""
        for i in range(self.degree, -1, -1):
            res += "%.3f" % (self.a[i]) + "x^" + str(i) + " "
        return res

    def calculate(self, x):
        res = 0
        for i in range(0, self.degree + 1):
            res += self.a[i] * math.pow(x, i)
        return res

    def plot(self, range_min=-1.0, range_max=1.0):
        xs = np.arange(range_min, range_max, 0.01)

        vfunc = np.vectorize(self.calculate)
        ys = vfunc(xs)

        print str(self)

        plt.figure(1)
        plt.title(str(self))
        plt.plot(xs, ys, 'k')
        plt.show()


class PeriodicPolynomial(Polynomial):
    """
    Polinomio periodico con periodo [0,1], con p(0) = p(1) = theta.
    Il valore del polinomio calcolato viene opportunamente modificato in modo che non sia mai negativo
    e che non siano presenti dei picchi troppo diversi da theta.
    """

    def __init__(self, degree, theta, build_mode='simple', points=2):
        # type: (int, float, str, int) -> None
        """
        Crea il polinomio
        :param degree: grado del polinomio
        :param theta: valore all'inizio e alla fine del periodo
        :param build_mode: `simple` o `fit`
        :param points: numero di punti in cui effettuare il fit (considerato solo se `build_mode=fit`)
        """
        super(PeriodicPolynomial, self).__init__(degree)
        self.theta = theta
        if build_mode == 'simple':
            # Aggiunge dei vincoli ai coefficienti per forzare la continuità.
            for i in range(0, degree + 1):
                self.a[i] = self.a[i] * degree
            # Vincoli per p(0) = theta e p(t) = theta
            self.a[0] = theta
            self.a[degree] = - sum(self.a[1:degree])
        elif build_mode == 'fit':
            # Crea dei punti a caso ed effettua il fit del polinomio
            xs = np.linspace(0, 1, points)
            ys = np.random.normal(loc=theta, scale=0.2, size=(points,))
            ys[0] = theta
            ys[points - 1] = theta
            c = P.polyfit(xs, ys, deg=degree)
            for i in range(0, degree + 1):
                self.a[i] = c[i]

    def raw_val(self, x):
        # type: (float) -> float
        """
        Valore grezzo del polinomio nel punto `x`
        :param x: punto in cui calcolare il valore
        :return: valore del polinomio
        """
        if x < 0:
            x = x + math.ceil(math.fabs(x))
        if x > 1:
            x = x - math.floor(x)
        return super(PeriodicPolynomial, self).calculate(x)

    def calculate(self, x):
        # type: (float) -> float
        """
        Valore scalato del polinomio nel punto `x`
        :param x: punto in cui calcolare il valore
        :return: valore del polinomio
        """
        if x < 0:
            x = x + math.ceil(math.fabs(x))
        if x > 1:
            x = x - math.floor(x)
        val = super(PeriodicPolynomial, self).calculate(x)
        val = math.fabs(val)
        if val > 10 * self.theta:
            val = math.log(val, 100)
        return val

    def plot(self, range_min=-1.0, range_max=1.0):
        xs = np.arange(range_min, range_max, 0.01)

        vfunc = np.vectorize(self.calculate)
        ys = vfunc(xs)

        print str(self)

        plt.figure(1)
        plt.title(str(self))
        plt.plot(xs, ys, 'k')
        plt.plot([range_min, range_max], [self.theta, self.theta], 'r--')
        plt.show()
