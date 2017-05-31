#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getopt, sys
import time

import datetime
import numpy as np
from sklearn.metrics import roc_auc_score

from generator.polynomial import Polynomial, PeriodicPolynomial


def main(argv):
    #poly = PeriodicPolynomial(grade=20, theta=1)
    #poly.plot(range_min=-1, range_max=1)

    poly2 = PeriodicPolynomial(degree=20, theta=1, build_mode='fit', points=15)
    poly2.plot(range_min=-1, range_max=1)


if __name__ == '__main__':
    m_start_time = time.time()
    m_start_cpu_time = time.clock()
    main(sys.argv[1:])
    print "-----------------------------"
    print "Total duration: ", time.time() - m_start_time, "seconds"
    print "CPU time:", time.clock() - m_start_cpu_time, "seconds"