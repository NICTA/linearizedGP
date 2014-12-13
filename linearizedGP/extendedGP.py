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

""" Extended Gaussian Process

    This module implements the extended Gaussian process for general non-linear
    likelihoods in the Gaussian process framework.

    Author:     Daniel Steinberg (daniel.steinberg@nicta.com.au)
    Institute:  NICTA
    Date:       17 Mar 2014

"""

import numpy as np
from linearizedGP.kernels import kern_se
from linearizedGP.gputils import jitchol, cholsolve, logdet
from linearizedGP.GP import GP


# The Extended Gaussian Process Class -----------------------------------------

class extendedGP(GP):

    def __init__(self, nlfunc, dnlfunc, kfunc=kern_se):
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
                dnlfunc: the derivative of nlfunc, i.e. g'(f). This has the
                    same input and output requirements as nlfunc.
                kfunc: the kernel function, look in the kernels module for
                    more kernel functions (defaults to square exponential).

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
        self.dnlfunc = dnlfunc

    def learn(self, x, y, kparams, ynoise, nlparams=None, dobj=1e-5,
              dparams=1e-8, maxit=200, verbose=False):
        """ Learn method for learning the posterior parameters, likelihood
            noise and kernel hyperparameters of the extended GP.

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

        return self._learn(self.__taylin, x, y, kparams, ynoise, nlparams,
                           dobj, dparams, maxit, verbose)

    def predict(self, xs):
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

    def __taylin(self, y, K, delta=1e-6, maxit=200, rate=0.75, maxsteps=25,
                 verbose=False):
        """ Posterior parameter learning using 1st order Taylor series
            expansion of the nonlinear forward model.

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
        Kchol = jitchol(K)

        # Bootstrap iterations
        obj = np.finfo(float).min
        endcond = "maxit"

        # Pre-allocate
        a = np.zeros(N)
        H = np.zeros((N, N))

        for i in range(maxit):

            # Store old values in case of divergence, and for "line search"
            objo, mo, ao, Ho = obj, m.copy(), a.copy(), H.copy()

            # Taylor series linearisation
            gm = self._passnlfunc(self.nlfunc, mo)
            a = self._passnlfunc(self.dnlfunc, mo)

            # Kalmain gain
            AK = (a * K).T
            AKAsig = a * AK
            AKAsig[np.diag_indices(N)] += self.ynoise**2
            H = cholsolve(jitchol(AKAsig), AK).T

            # Do a bit of a heuristic line search for best step length
            for j in range(maxsteps):

                step = rate**j
                m = (1 - step) * mo + step * H.dot(y - gm + a * mo)

                # MAP objective
                ygm = y - self._passnlfunc(self.nlfunc, m)
                obj = -0.5 * (m.T.dot(cholsolve(Kchol, m))
                              + (ygm**2).sum()/self.ynoise**2)

                if obj >= objo:
                    break

            dobj = abs((objo - obj) / obj)

            # Divergence, use previous result
            if (obj < objo) & (dobj > delta):
                m, obj, a, H = mo, objo, ao, Ho  # recover old values
                endcond = "diverge"
                break

            # Convergence, use latest result
            if dobj <= delta:
                endcond = "converge"
                break

        if verbose is True:
            print("iters: {}, endcond = {}, MAP = {}, "
                  "delta = {:.2e}".format(i, endcond, obj, dobj))

        # Useful equations for Free energy
        C = K - H.dot((a * K).T)
        gm = self._passnlfunc(self.nlfunc, m)

        # Calculate Free Energy
        Feng = -0.5 * (N * np.log(np.pi * 2 * self.ynoise**2)
                       + ((y - gm)**2).sum() / self.ynoise**2
                       + m.T.dot(cholsolve(Kchol, m)) - logdet(C, dochol=True)
                       + logdet(Kchol))

        return m, C, Feng
