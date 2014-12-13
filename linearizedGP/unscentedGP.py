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

""" Unscented Gaussian Process

    This module implements the unscented linearised Gaussian process for
    general non-linear likelihoods in the Gaussian process framework.

    Author:     Daniel Steinberg (daniel.steinberg@nicta.com.au)
    Institute:  NICTA
    Date:       17 Mar 2014

"""

import numpy as np
from linearizedGP.kernels import kern_se
from linearizedGP.gputils import jitchol, cholsolve, logdet
from linearizedGP.GP import GP


# The Unscented Gaussian Process Class ----------------------------------------

class unscentedGP(GP):

    def __init__(self, nlfunc, kfunc=kern_se, kappa=0.5):
        """ The unscented Gaussian Process class for arbitrary forward models

            p(y|X) = int N(y|g(f), ynoise**2*I) GP(f|0, K(X,X)) df

            where g(.) is a forward model that transforms the latent function,
            f. g(.) can be nonlinear and non-differentiable. The posterior,
            p(y|X) is learned by linearising the nonlinear formward model g(.)
            about the posterior mean using *statistical linearisation*. Also,
            the kernel hyperparameters and ynoise using derivative free methods
            from the NLopt library (BOBYQA).

            Arguments:
                nlfunc: the non-linear function g in, y = g(f) + e, that can
                    take arrays of shape (N,) or (N, I) as input and return
                    arrays of the same dimension on the output. Where shape(f)
                    is (N,) and I is I samples of f.
                kfunc: the kernel function, look in the kernels module for
                    more kernel functions (defaults to square exponential).
                kappa: [optional] the parameter that controls the spread of
                    sigma points in the unscented transform and their
                    contribution to the linearisation. 0.5 results in uniform
                    weighting and a moderate spread, higher numbers result in
                    larger spread and more heavily weigthing the mean sigma
                    point.

            Returns:
                An instance of the unscentedGP class.

            Note:
                Also see the
                - learn()
                - predict()
                Methods for learning this GP and also for predicting E[y*] for
                new inputs, x*. Also see
                - learnLB()
                - learnUB()
                for placing lower and upper bounds on the hyperparameters.

        """

        GP.__init__(self, kfunc)

        # Functions and their parameters
        self.nlfunc = nlfunc

        # Sigma point parameters
        self.kappa = kappa
        self.W = self.__sigweights(1)

    def learn(self, x, y, kparams, ynoise, nlparams=None, dobj=1e-5,
              dparams=1e-8, maxit=200, verbose=False):
        """ Learn method for learning the posterior parameters, likelihood
            noise and kernel hyperparameters of the unscented GP.

            Arguments:
                x: [DxN] array of N input samples with a dimensionality of D.
                y: N array of training outputs (dimensionality of 1)
                kparams: a tuple of initial values corresponding to the kernel
                    hyperparameters of the kernel function input to the
                    constructor.
                ynoise: a scalar initial value for the observation (y) noise.
                nlparams: [optional] initial parameters to be passed to the
                    forward model, these are also optimised by NLopt (scalar or
                    small vector)
                dobj: [optional] the convergence threshold for the objective
                    function (free energy) used by NLopt.
                dparams: [optional] the convergence threshold for the
                    hyperparameter values.
                maxit: [optional] maximum number of iterations for learning the
                    hyperparameters.
                verbose: [optional] whether or not to display current learning
                    progress to the terminal.

            Returns:
                The final objective function value (free energy). Also
                internally all of the final parameters and hyperparameters are
                stored.

            Note:
                This stops learning as soon as one of the convergence criterion
                is met (objective, hyperparameters or iterations).

        """

        return self._learn(self.__statlin, x, y, kparams, ynoise, nlparams,
                           dobj, dparams, maxit, verbose)

    def predict(self, xs):
        """ Prediction of m* (the posterior mean of the latent function) and
            E[y*] at test points, x*, using the unscented transform to evaluate
            E[y*].

            Arguments:
                xs: [DxN] test points for prediction

            Returns:
                Eys: array of N predictions of E[y*]
                Vms: array of N predictions of the variance of y*, V[y*].
                Ems: array of N predictions of m*
                Vms: array of N predictions of the variance of m*, V[m*].
        """

        # Check we have trained and the inputs
        D, N = self._check_prediction_inputs(xs)

        # Evaluate test kernel vectors and do a stable inversion
        ks = self.kfunc(self.x, np.atleast_2d(xs), *self.kparams)
        Kinvks = cholsolve(self.Kchol, ks)

        # Pre-allocate
        Ems, Vms, Eys, Vys = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

        for n, xn in enumerate(xs.T):

            # Evaluate the test kernel vectors
            kss = self.kfunc(np.atleast_2d(xn).T, np.atleast_2d(xn).T,
                             *self.kparams)

            # Predict the latent function
            Ems[n] = (Kinvks[:, n].T).dot(self.m)
            Vms[n] = kss - Kinvks[:, n].T.dot(ks[:, n]
                                              - self.C.dot(Kinvks[:, n]))

            # Use the UT to predict the target value
            Ms = self.__makesigmas1D(Vms[n])
            Ms += Ems[n]
            Ys = self._passnlfunc(self.nlfunc, Ms)

            Eys[n] = (self.W * Ys).sum()
            Vys[n] = (self.W * (Ys - Eys[n])**2).sum()

        return Eys, Vys, Ems, Vms

    def quadpredict(self, xs):
        """ Prediction of m* (the posterior mean of the latent function) and
            E[y*] at test points, x*, using quadrature to evaluate E[y*].

            Arguments:
                xs: [DxN] test points for prediction

            Returns:
                Eys: array of N predictions of E[y*]
                eEys: array of N errors on each E[y*] integral evaluation
                Ems: array of N predictions of m*
                Vms: array of N predictions of the variance of m*, V[m*].
        """

        return self._quadpredict(xs)

    def __statlin(self, y, K, delta=1e-6, maxit=200, rate=0.75, maxsteps=25,
                  verbose=False):
        """ Posterior parameter learning using the unscented transform and
            statistical linearisation.

            Arguments:
                y: N array of training outputs (dimensionality of 1)
                K: the [NxN] covariance matrix with the current hyper parameter
                    estimates.
                delta: [optional] the convergence threshold for the objective
                    function (free energy).
                dparams: [optional] the convergence threshold for the
                    hyperparameter values.
                maxit: [optional] maximum number of iterations for learning the
                    posterior parameters.
                rate: [optional] the learning rate of the line search used in
                    each of the Gauss-Newton style iterations for the posterior
                    mean.
                maxsteps: [optional] the maximum number of line-search steps.
                verbose: [optional] whether or not to display current learning
                    progress to the terminal.

            Returns:
                The final objective function value (free energy), and the
                posterior mean (m) and Covariance (C).
        """

        # Establish some parameters
        N = y.shape[0]
        m = np.random.randn(N) / 100.0

        # Make sigma points in latent space
        C = K.copy()
        Kchol = jitchol(K)
        Ms = self.__makesigmas1D(K)

        # Bootstrap iterations
        obj = np.finfo(float).min
        endcond = "maxit"
        ybar = y.copy()

        for i in range(maxit):

            # Store old values in case of divergence, and for "line search"
            objo, mo, ybaro = obj, m.copy(), ybar.copy()

            # Sigma points in obs. space and sigma point stats
            Ys = self._passnlfunc(self.nlfunc, m[:, np.newaxis] + Ms)
            ybar = (self.W * Ys).sum(axis=1)
            Sym = (self.W * (Ys - ybar[:, np.newaxis]) * Ms).sum(axis=1)

            # Statistical linearisation
            a = Sym / C.diagonal()

            # Kalmain gain
            AK = (a * K).T
            AKAsig = a * AK
            AKAsig[np.diag_indices(N)] += self.ynoise**2
            H = cholsolve(jitchol(AKAsig), AK).T

            # Do a bit of a heuristic line search for best step length
            for j in range(maxsteps):

                step = rate**j
                m = (1 - step) * mo + step * H.dot(y - ybar + a * mo)

                # MAP objective
                ygm = y - self._passnlfunc(self.nlfunc, m)
                obj = -0.5 * (m.T.dot(cholsolve(Kchol, m))
                              + (ygm**2).sum()/self.ynoise**2)

                if obj >= objo:
                    break

            dobj = abs((objo - obj) / obj)

            # Divergence, use previous result
            if (obj < objo) & (dobj > delta):
                m, ybar, obj = mo, ybaro, objo  # recover old values
                endcond = "diverge"
                break

            # Make posterior C if m has not diverged
            C = K - H.dot(AK)
            Ms = self.__makesigmas1D(C)

            # Convergence, use latest result
            if dobj <= delta:
                endcond = "converge"
                break

        if verbose is True:
            print("iters: {}, endcond = {}, MAP = {}, "
                  "delta = {:.2e}".format(i, endcond, obj, dobj))

        # Calculate Free Energy
        Feng = -0.5 * (N * np.log(np.pi * 2 * self.ynoise**2)
                       + ((y - ybar)**2).sum() / self.ynoise**2
                       + m.T.dot(cholsolve(Kchol, m)) - logdet(C, dochol=True)
                       + logdet(Kchol))

        return m, C, Feng

    def __makesigmas1D(self, K):
        """ Make zero centred sigma points in the latent space using the
            diagonal of K only.
        """

        N = 1 if np.isscalar(K) else K.shape[0]

        if N == 1:
            sqrtK = np.sqrt(K)
            zero = 0
        else:
            sqrtK = np.sqrt(K.diagonal())
            zero = np.zeros(N)

        sqrtsK = np.sqrt(1 + self.kappa) * sqrtK
        Fs = np.vstack((zero, sqrtsK, -sqrtsK)).T

        return Fs

    def __sigweights(self, N):
        """ Make the sigma point weights. """

        W = np.concatenate(([self.kappa / (N + self.kappa)],
                            [1.0 / (2 * (N + self.kappa))] * (2 * N)))

        return W
