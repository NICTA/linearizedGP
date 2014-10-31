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

""" Run the USPS handwritten digits experiment from our NIPS 2014 paper. Also
    see uspsbclass.m for the octave/matlab algorithms. You'll need scikit
    learn for this.

    Author:     Daniel Steinberg (daniel.steinberg@nicta.com.au)
    Institute:  NICTA
    Date:       4 Sep 2014

"""

import numpy as np
from linearizedGP import unscentedGP
from linearizedGP import extendedGP
from linearizedGP import kernels
import scipy.io as sio
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# Settings --------------------------------------------------------------------

# Data location
datapath = "data/USPS_3_5_data.mat"

# Classification properties
kbounds = (0.1, 0.1)
kinit = (1.0, 1.0)
nbound = 1e-7
ninit = 1e-7

# Sigmoid functions
lgsig = lambda f: 1.0 / (1 + np.exp(-f))
dlgsig = lambda f: lgsig(f) * lgsig(-f)

# kernel functions
kfunc = kernels.kern_selog


# Data ------------------------------------------------------------------------

USPSdata = sio.loadmat(datapath, squeeze_me=True)

x = USPSdata['x'].T
xs = USPSdata['xx'].T
y = USPSdata['y']
ys = USPSdata['yy']
y[y == -1] = 0
ys[ys == -1] = 0


# Train the non-GP classifiers ------------------------------------------------

print "\nLearning the support vector classifier"
C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(y=y, n_folds=3)
grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid=param_grid,
                    cv=cv, verbose=1)
grid.fit(x.T, y)
svc = grid.best_estimator_

print "\nLearning the logistic regression classifier"
lreg = LogisticRegression(penalty='l2')
lreg.fit(x.T, y)


# Train the GPs ---------------------------------------------------------------

# Statistical linearisation
print "\nLearning statistically linearised classifier"
sgp = unscentedGP.unscentedGP(nlfunc=lgsig, kfunc=kfunc)
sgp.learnLB(np.log(kbounds), ynoise=nbound)
lml = sgp.learn(x, y, np.log(kinit), ynoise=ninit, verbose=True)

print "Log marginal likelihood = {0}".format(lml)
print "Hyper-parameters = {0}, noise = {1}".format(sgp.kparams, sgp.ynoise)

# Taylor linearisation
print "\nLearning Taylor series linearised classifier"
tgp = extendedGP.extendedGP(nlfunc=lgsig, dnlfunc=dlgsig, kfunc=kfunc)
tgp.learnLB(np.log(kbounds), ynoise=nbound)
lml = tgp.learn(x, y, np.log(kinit), ynoise=ninit, verbose=True)

print "Log marginal likelihood = {0}".format(lml)
print "Hyper-parameters = {0}, noise = {1}".format(tgp.kparams, tgp.ynoise)


# Prediction ------------------------------------------------------------------

def bernloglike(pys):
    return -(ys * np.log(pys) + (1 - ys) * np.log(1 - pys)).mean()


def errrate(pys):
    return float((ys != (pys >= 0.5)).sum()) / ys.shape[0]

print "\n\nResults: \n----------------"

# Statlin
pys_s, epys_s, Ems_s, Vms_s = sgp.quadpredict(xs)
print "Stat lin: av nll = {:.6f}, Error rate = {:.6f}"\
      .format(bernloglike(pys_s), errrate(pys_s))

# Taylorlin
pys_t, epys_t, Ems_t, Vms_t = tgp.predict(xs)
print "Tayl lin: av nll = {:.6f}, Error rate = {:.6f}"\
      .format(bernloglike(pys_t), errrate(pys_t))

# SVM
pys_v = svc.predict_proba(xs.T)[:, 1]
print "SVM: av nll = {:.6f}, Error rate = {:.6f}"\
      .format(bernloglike(pys_v), errrate(pys_v))

# Logistic Regression
pys_r = lreg.predict_proba(xs.T)[:, 1]
print "Logistic: av nll = {:.6f}, error rate = {:.6f}"\
      .format(bernloglike(pys_r), errrate(pys_r))
