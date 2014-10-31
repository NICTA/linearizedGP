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
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with linearizedGP. If not, see <http://www.gnu.org/licenses/>.

""" Run some *toy* experiments on the nonlinear GPs. This uses data generated
    from datagen.py. This is the script used for experiment one in the NIPS
    2014 submission.

    Author:     Daniel Steinberg (daniel.steinberg@nicta.com.au)
    Institute:  NICTA
    Date:       4 Sep 2014

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from linearizedGP import unscentedGP
from linearizedGP import extendedGP
from linearizedGP import GP
from linearizedGP import kernels


# Some parameters for the experiment ------------------------------------------

#dataset = 'data/tanhdata.mat'
#dataset = 'data/sindata.mat'
#dataset = 'data/lineardata.mat'
#dataset = 'data/poly3data.mat'
dataset = 'data/expdata.mat'

plot = False
saveresults = True
dolinear = False

# -----------------------------------------------------------------------------

# Results save name
splitname = (dataset.replace('data/', 'results/')).split('.')
savename = ''.join(splitname[:-1]) + '_res.' + splitname[-1]
restext = ''.join(splitname[:-1]) + '_res.txt'


# Load and convert the dataset
data = sio.loadmat(dataset, squeeze_me=True)
nlfunc = lambda f: eval(data['func'])
dnlfunc = lambda f: eval(data['dfunc'])
folds = len(data['test'])


# Make the GPs
stlgp = unscentedGP.unscentedGP(nlfunc, kernels.kern_m52)
tlgp = extendedGP.extendedGP(nlfunc, dnlfunc, kernels.kern_m52)

stlgp.learnLB((1e-1, 1e-1), ynoise=1e-2)
tlgp.learnLB((1e-1, 1e-1), ynoise=1e-2)

if dolinear is True:
    lgp = GP.GP(kernels.kern_m52)
    lgp.learnLB((1e-1, 1e-1), ynoise=1e-2)

# Pre-allocate
llf_s = np.zeros(folds)
llf_t = np.zeros(folds)
SMSEf_s = np.zeros(folds)
SMSEf_t = np.zeros(folds)
SMSEy_s = np.zeros(folds)
SMSEy_t = np.zeros(folds)

resdic = {
    'Em_s': [],
    'Vm_s': [],
    'Ey_s': [],
    'Em_t': [],
    'Vm_t': [],
    'Ey_t': []
    }

if dolinear is True:
    llf_l = np.zeros(folds)
    SMSEf_l = np.zeros(folds)

    resdic['Em_l'] = []
    resdic['Vm_l'] = []


def nll(x, Ex, Vx, N):
    return 0.5 * (((x - Ex)**2 / Vx + np.log(2 * np.pi * Vx))).sum() / N


def smse(x, Ex, N, sig):
    return ((x - Ex)**2).sum() / (N * sig)


# Do the experiment
for k in xrange(folds):

    xr = data['x'][data['train'][k, :].astype(int)]
    yr = data['y'][data['train'][k, :].astype(int)]
    xs = data['x'][data['test'][k, :].astype(int)]
    ys = data['y'][data['test'][k, :].astype(int)]
    fs = data['f'][data['test'][k, :].astype(int)]

    Ntrain = xr.shape[0]
    Ntest = xs.shape[0]
    fvartest = fs.var()
    yvartest = ys.var()

    print "Fold {0}".format(k)

    Feng = stlgp.learn(xr, yr, (1, 1), ynoise=1, verbose=False)
    Eys_s, eEys_s, Ems_s, Vms_s = stlgp.quadpredict(xs)
    SMSEy_s[k] = smse(ys, Eys_s, Ntest, yvartest)
    SMSEf_s[k] = smse(fs, Ems_s, Ntest, fvartest)
    llf_s[k] = nll(fs, Ems_s, Vms_s, Ntest)

    print "\tStat. lin. GP F = {:.5f}, ll(f) = {:.5f}, smse(f) = {:.5f},"\
        " smse(y) = {:.5f}"\
        .format(Feng, llf_s[k], SMSEf_s[k], SMSEy_s[k])
    print "\thyperparams = {}, lstd = {:.5f}.".format(stlgp.kparams,
                                                      stlgp.ynoise)

    lml_t = tlgp.learn(xr, yr, (1, 1), ynoise=1, verbose=False)
    Eys_t, eEys_t, Ems_t, Vms_t = tlgp.predict(xs)
    SMSEy_t[k] = smse(ys, Eys_t, Ntest, yvartest)
    SMSEf_t[k] = smse(fs, Ems_t, Ntest, fvartest)
    llf_t[k] = nll(fs, Ems_t, Vms_t, Ntest)

    print "\tTay. lin. GP F = {:.5f}, ll(f) = {:.5f}, smse(f) = {:.5f},"\
        " smse(y) = {:.5f}"\
        .format(lml_t, llf_t[k], SMSEf_t[k], SMSEy_t[k])
    print "\thyperparams = {}, lstd = {:.5f}.".format(tlgp.kparams,
                                                      tlgp.ynoise)

    if dolinear is True:
        lml_l = lgp.learn(xr, yr, (1, 1), ynoise=1, verbose=False)
        Ems_l, Vms_l = lgp.predict(xs)
        SMSEf_l[k] = smse(fs, Ems_l, Ntest, fvartest)
        llf_l[k] = nll(fs, Ems_l, Vms_l, Ntest)

        print "\tLin. GP F = {:.5f}, ll(f) = {:.5f}, smse(f) = {:.5f}"\
            .format(lml_l, llf_l[k], SMSEf_l[k])
        print "\thyperparams = {}, lstd = {:.5f}.".format(lgp.kparams,
                                                          lgp.ynoise)

    if plot is True:
        plt.plot(data['x'], data['f'], 'k', xs, Ems_s, 'bo', xs, Ems_t, 'ro')
        plt.plot(data['x'], data['y'], 'k--', xs, Eys_s, 'b.', xs, Eys_t, 'r.')
        plt.legend(['f', 'E[m] st', 'E[m] ta', 'y', 'E[y] st', 'E[y] ta'])
        plt.grid(True)
        plt.show()

    if saveresults is True:
        resdic['Em_s'].append(Ems_s)
        resdic['Vm_s'].append(Vms_s)
        resdic['Ey_s'].append(Eys_s)
        resdic['Em_t'].append(Ems_t)
        resdic['Vm_t'].append(Vms_t)
        resdic['Ey_t'].append(Eys_t)

        if dolinear is True:
            resdic['Em_l'].append(Ems_l)
            resdic['Vm_l'].append(Vms_l)

resstr = "Final result:" \
         "\n\t Statistical linearisation --" \
         "\n\t\t -ll(f): av = {:.5f}, std = {:.5f}" \
         "\n\t\t smse(f): av = {:.4e}, std = {:.4e}" \
         "\n\t\t smse(y): av = {:.4e}, std = {:.4e}" \
         "\n\t Taylor linearisation  --" \
         "\n\t\t -ll(f): av = {:.5f}, std = {:.5f}" \
         "\n\t\t smse(f): av = {:.4e}, std = {:.4e}"\
         "\n\t\t smse(y): av = {:.4e}, std = {:.4e}" \
         .format(llf_s.mean(), llf_s.std(),
                 SMSEf_s.mean(), SMSEf_s.std(),
                 SMSEy_s.mean(), SMSEy_s.std(),
                 llf_t.mean(), llf_t.std(),
                 SMSEf_t.mean(), SMSEf_t.std(),
                 SMSEy_t.mean(), SMSEy_t.std())


if dolinear is True:
    resstr += "\n\t Linear GP --" \
              "\n\t\t -ll(f): av = {:.5f}, std = {:.5f}" \
              "\n\t\t smse(f): av = {:.4e}, std = {:.4e}" \
              .format(llf_l.mean(), llf_l.std(),
                      SMSEf_l.mean(), SMSEf_l.std())

resstr += "\n\n"
resstr += "& UGP & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\\n"\
    .format(llf_s.mean(), llf_s.std(),
            SMSEf_s.mean(), SMSEf_s.std(),
            SMSEy_s.mean(), SMSEy_s.std())
resstr += "& EGP & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\\n"\
    .format(llf_t.mean(), llf_t.std(),
            SMSEf_t.mean(), SMSEf_t.std(),
            SMSEy_t.mean(), SMSEy_t.std())
if dolinear is True:
    resstr += "& GP & {:.5f} & {:.5f} & {:.5f} & {:.5f} & -- & -- \\\\\n"\
        .format(llf_l.mean(), llf_l.std(),
                SMSEf_l.mean(), SMSEf_l.std())

print "\n\n" + resstr


# Save results
if saveresults is True:

    with open(restext, "w") as resfile:
        resfile.write(data['func'] + ":\n\n" + resstr)

    resdic['Em_s'] = np.array(resdic['Em_s'])
    resdic['Vm_s'] = np.array(resdic['Vm_s'])
    resdic['Ey_s'] = np.array(resdic['Ey_s'])
    resdic['Em_t'] = np.array(resdic['Em_t'])
    resdic['Vm_t'] = np.array(resdic['Vm_t'])
    resdic['Ey_t'] = np.array(resdic['Ey_t'])

    if dolinear is True:
        resdic['Em_l'] = np.array(resdic['Em_l'])
        resdic['Vm_l'] = np.array(resdic['Vm_l'])

    sio.savemat(savename, resdic)
