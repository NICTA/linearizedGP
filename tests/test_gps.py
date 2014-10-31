#! /usr/bin/env python

# linearizedGP -- Implementation of extended and unscented Gaussian processes.
# Copyright (C) 2014 National ICT Australia (NICTA)
#
# This file is part of linearizedGP.
#
# linearizedGP is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# linearizedGP is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with linearizedGP. If not, see <http://www.gnu.org/licenses/>.

""" Test script for running the EGP, UGP and a regular GP.

    Author:       Daniel Steinberg (daniel.steinberg@nicta.com.au)
    Institute:    NICTA
    Date:         29 Jun 2014

"""

import time
import numpy as np
from linearizedGP import unscentedGP
from linearizedGP import extendedGP
from linearizedGP import GP
from linearizedGP import kernels
import matplotlib.pyplot as plt
import pickle as pk


# Some parameters for the dataset ---------------------------------------------
npoints = 200   # Training points
ppoints = 1000  # Prediction points
noise = 0.2     # Likelihood noise for generated data
k_sigma = 0.8   # Kernel parameters for the random GP draw
k_length = 0.6

# GP settings
kfunc = kernels.kern_m52
gptype = 'UGP'
#gptype = 'EGP'
#gptype = 'GP'


# Forward models --------------------------------------------------------------

# Non-differentiable (only works with the UGP)
nlfunc = lambda f: 2 * np.sign(f) + f**3
#nlfunc = lambda f: np.round(np.exp(2 * f))

# Differentiable
#nlfunc = lambda f: f
#dnlfunc = lambda f: np.ones(f.shape)
#nlfunc = lambda f: np.tanh(2*f)
#dnlfunc = lambda f: 2 - 2*np.tanh(f)**2
#nlfunc = lambda f: f**3 + f**2 + f
#dnlfunc = lambda f: 3*f**2 + 2*f + 1
#nlfunc = lambda f: np.exp(f)
#dnlfunc = lambda f: np.exp(f)
#nlfunc = lambda f: np.sin(f)
#dnlfunc = lambda f: np.cos(f)
#nlfunc = lambda f: unscentedGP.logsig(f)


# Latent function and data generation -----------------------------------------

# Draw from a GP
x = np.linspace(-2*np.pi, 2*np.pi, ppoints + npoints)
U, S, V = np.linalg.svd(kfunc(x[np.newaxis, :], x[np.newaxis, :], k_sigma,
                        k_length))
L = U.dot(np.diag(np.sqrt(S))).dot(V)
f = np.random.randn(ppoints + npoints).dot(L)
y = nlfunc(f) + np.random.randn(npoints + ppoints) * noise

# Training data
tind = np.zeros(ppoints + npoints).astype(bool)
tind[np.random.randint(0, ppoints + npoints, npoints)] = True
xt = x[tind]
ft = f[tind]
yt = y[tind]

# Test data
sind = ~ tind
xs = x[sind]
fs = f[sind]
ys = y[sind]


# Create, learn and predict from the GPs --------------------------------------
if gptype is 'UGP':
    gp = unscentedGP.unscentedGP(nlfunc, kfunc)
elif gptype is 'EGP':
    gp = extendedGP.extendedGP(nlfunc, dnlfunc, kfunc)
elif gptype is 'GP':
    gp = GP.GP(kfunc)
else:
    raise ValueError('invalid GP type')

# Learn GP
start = time.clock()

gp.learnLB((1e-1, 1e-1), ynoise=1e-2)
lml = gp.learn(xt, yt, (1, 1), ynoise=1, verbose=True)

elapsed = (time.clock() - start)

print "\nTraining time = {0} sec".format(elapsed)
print "Free Energy = {0}".format(lml)
print "Hyper-parameters = {0}, noise = {1}".format(gp.kparams, gp.ynoise)
print "Non-linear function parameters = {0}".format(gp.nlparams)

# Predict
start = time.clock()
if gptype is 'UGP':
    Eys, Vys, Ems, Vms = gp.predict(xs)
elif gptype is 'EGP':
    Eys, _, Ems, Vms = gp.predict(xs)
elif gptype is 'GP':
    Ems, Vms = gp.predict(xs)
    Eys = Ems
    Vys = Vms
elapsed = (time.clock() - start)

print "Prediction time = {0} sec".format(elapsed)


# Performance evaluation ------------------------------------------------------

if gptype is ('UGP' or 'GP'):
    heldoutp = -0.5 * ((ys - Eys)**2 / Vys + np.log(2 * np.pi * Vys))

RMSE_y = np.sqrt(((ys - Eys)**2).sum() / ppoints)
print "Target prediction root mean square error: {0}".format(RMSE_y)

if gptype is ('UGP' or 'GP'):
    heldoutp = -0.5 * ((ys - Eys)**2 / Vys + np.log(2 * np.pi * Vys))
    print "Target prediction log likelihood: {0}".format(heldoutp.sum())

heldoutfs = -0.5 * ((fs - Ems)**2 / Vms + np.log(2 * np.pi * Vms))
print "Latent prediction log likelihood: {0}".format(heldoutfs.sum())


# Plot the results ------------------------------------------------------------

plt.figure(figsize=(11, 6))
ax = plt.subplot(111)
plt.plot(xs, fs, label='True process, $f_{true}$', linewidth=2)
plt.plot(xt, yt, 'k.', label='Training data $\{\mathbf{x}_{n}, y_{n}\}$')
plt.plot(xs, Ems, 'r--', label='Estimated process, $m^*$', linewidth=2)
Sfs = 2 * np.sqrt(Vms)
plt.fill_between(xs, Ems + Sfs, Ems - Sfs, facecolor='red', edgecolor='red',
                 alpha=0.3, label=None)
plt.plot(xs, Eys, 'g', label='Predictions, $\langle y^*\!\\rangle_{qf^*}$',
         linewidth=2)
if gptype is ('UGP' or 'GP'):
    ss = 2 * np.sqrt(Vys)
    plt.fill_between(xs, Eys + ss, Eys - ss, facecolor='green',
                     edgecolor='green', alpha=0.3, label=None)
plt.xlabel('inputs ($\mathbf{x}$)', fontsize=20)
plt.ylabel('targets ($y$) and latent function ($f$)',
           fontsize=20)
plt.legend(loc=3, fontsize=18)
plt.autoscale(tight=True)
ax.set_yticklabels(ax.get_yticks(), size=16)
ax.set_xticklabels(ax.get_xticks(), size=16)
plt.grid(True)

with open('test.pk', 'wb') as f:
    pk.dump(ax, f)

plt.savefig('test.pdf', bbox_inches='tight')

plt.show()
