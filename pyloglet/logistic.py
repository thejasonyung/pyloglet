#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Marchetti, C.: Society as a Learning System.
# Technological Forecasting and Social Change, 18, 267–282 (1980)
# t : time t for which we are evaluating
# k : saturation
# tm : midpoint
# dt : range from 10%-50% or 50%-90% saturation
def logistic(t, k, tm, dt):
    return k / ( 1 + np.exp( -(np.log(81) / dt) * ( t - tm ) ) )


class levenberg_marquardt(object):
    def __init__(self, p0, y0=0, hold=False):
        """
        p0: initial parameters
        y0: initial displacement
        hold: boolean flags to keep parameters constant
        """
        self.p0 = p0
        self.y0 = y0
        self.hold = hold or [False for x in p0]

    def lm_logistic(self, t, *p):
        """
        Take the initial displacement (y0)
        and add in each logistic
        """
        y = self.y0
        i = 0
        while len(self.p0) > i:
            k = self.p0[i + 0] if self.hold[i + 0] else p[i + 0]  # hold using p[0]
            tm = np.full(len(t), self.p0[i + 1]) if self.hold[i + 1] else p[i + 1] # hold using np.full(len(t), 42)
            dt = self.p0[i + 2] if self.hold[i + 2] else p[i + 2] # hold using p[2]
            y += logistic(t, k, tm, dt)
            i += 3
        return y

    def fit(self, xdata, ydata):
        # From the SciPy documentation
        # (http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit)
        # popt : array
        # Optimal values for the parameters so that the sum of the squared error of f(xdata, *popt) - ydata is minimized
        # pcov : 2d array
        # The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        # To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
        popt, pcov = curve_fit(self.lm_logistic, xdata, ydata, self.p0)
        return popt

class marchetti(object):
    def __init__(self):
        """
        """
        pass

    def linear(self, t, p0, p1):
        pass

    def fisher_pry(self, k, y):
        normalized = y / k
        return np.log( normalized / (1 - normalized) )

    def cm_fit(self, k, xdata, ydata):
        # filter out anything below 0 and above k
        filtered = filter(lambda xy: xy[1] > 0 and xy[1] < k, zip(xdata, ydata))
        xfiltered, yfiltered = zip(*filtered)
        slope, intercept, r_value, p_value, std_err = stats.linregress(xfiltered, map(lambda y: self.fisher_pry(k,y), yfiltered))
        return - intercept / slope, np.log(81) / slope, r_value

    def fit(self, xdata, ydata):
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
        # For each logistic:
        #   Select some K values
        #     Do Fisher-Pry transform
        #   Find the K that gives us the best least square linear regression (best R)
        #   Map from a + bx to dt and tm, respectively
        k0 = ydata[-1]
        delta = 0.1 * k0;

        # init run
        dt, tm, r_value = self.cm_fit(k0 - delta, xdata, ydata)
        best_fit = r_value
        best_params = [k0, dt, tm]
        
        for k in np.linspace(k0-delta, k0+delta):
            dt, tm, r_value = self.cm_fit(k, xdata, ydata)
            if np.abs(r_value) > np.abs(best_fit):
                best_fit = r_value
                best_params = [k, dt, tm]

        return best_params
