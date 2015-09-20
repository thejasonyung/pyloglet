#!/usr/bin/python
# -*- coding: utf-8 -*-
from pyloglet import logistic

# Sunflower data (single logistic)
# data
xdata = [ 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84 ]
ydata = [ 0.00, 17.93, 36.36, 67.76, 98.10, 131.00, 169.50, 205.50, 228.30, 247.10, 250.50, 253.80, 254.50 ]
# initial guesses
p0 = [ 257, 42, 38 ]

popt = logistic.levenberg_marquardt(p0).fit(xdata, ydata)

print popt

# hold K
popt = logistic.levenberg_marquardt(p0, hold=[True, False, False]).fit(xdata, ydata)

print popt

# hold K
popt = logistic.marchetti().fit(xdata, ydata)

print popt
