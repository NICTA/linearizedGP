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

""" Generate random datasets for testing the nonlinear GPs (used in the NIPS
    2014 submission).

    Author:     Daniel Steinberg (daniel.steinberg@nicta.com.au)
    Institute:  NICTA
    Date:       4 Sep 2014

"""

import os
from linearizedGP import gputils, kernels
import numpy as np
import scipy.io as sio


# Some parameters for the dataset
npoints = 1000  # Testing and training points
noise = 0.2
folds = 5
plot = False
kfunc = kernels.kern_m52
k_sigma = 0.8
k_length = 0.6

savedir = 'data'

# Nonlinear functions

#savename = "signdata.mat"
#fctn = "2 * np.sign(f) + f**3"
#dfctn = ""

#savename = "tanhdata.mat"
#fctn = "np.tanh(2*f)"
#dfctn = "2 - 2 * np.tanh(2*f)**2"

#savename = "sindata.mat"
#fctn = "np.sin(f)"
#dfctn = "np.cos(f)"

#savename = "lineardata.mat"
#fctn = "f"
#dfctn = "np.ones(f.shape)"

#savename = 'poly3data.mat'
#fctn = "f**3 + f**2 + f"
#dfctn = "3*f**2 + 2*f + 1"

savename = "expdata.mat"
fctn = "np.exp(f)"
dfctn = "np.exp(f)"

nlfunc = lambda f: eval(fctn)
dnlfunc = lambda f: eval(dfctn)

# Construct the dataset
x = np.linspace(-2 * np.pi, 2 * np.pi, npoints)
fseed = np.random.randn(npoints)
U, S, V = np.linalg.svd(kfunc(x[np.newaxis, :], x[np.newaxis, :], k_sigma,
                        k_length))
L = U.dot(np.diag(np.sqrt(S))).dot(V)
f = fseed.dot(L)
y = nlfunc(f) + np.random.randn(npoints) * noise


# Make the dictionary to save into a mat structure
datadic = {
    'noise':        noise,
    'func':         fctn,
    'dfunc':        dfctn,
    'x':            x,
    'f':            f,
    'y':            y,
    'train':        [],
    'test':         []
    }

# Save the data to disk
if not os.path.exists(savedir):
    os.mkdir(savedir)

for k, (sind, rind) in enumerate(gputils.k_fold_CV_ind(npoints, k=folds)):

    datadic['train'].append(rind)
    datadic['test'].append(sind)

datadic['train'] = np.array(datadic['train'])
datadic['test'] = np.array(datadic['test'])

sio.savemat(os.path.join(savedir, savename), datadic)
