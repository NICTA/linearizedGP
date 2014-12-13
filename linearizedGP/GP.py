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

""" Gaussian Process Base Class

    This is the GP base class for the nonlinear GPs. This also implements a
    "vanilla" linear GP.

    Author:     Daniel Steinberg (daniel.steinberg@nicta.com.au)
    Institute:  NICTA
    Date:       28 Sep 2014

"""

import numpy as np
import nlopt
from linearizedGP.kernels import kern_se
from linearizedGP.gputils import jitchol, cholsolve, logdet
import scipy.integrate as spint


class GP(object):

    def __init__(self, kfunc=kern_se):
        """ The base Gaussian Process class, which also implements a basic GP.

            y ~ GP(0, C),

            where,

            C_ij = k(x_i, x_j|theta) + del_i=j * ynoise**2,

            and the learn() method learns theta and ynoise using derivative
            free methods from the NLopt library (BOBYQA).

            Arguments:
                kfunc: the kernel function, look in the kernels module for
                    more kernel functions (defaults to square exponential).

            Returns:
                An instance of the GP class.

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

        # Kernel Functions and Nonlinearities
        self.kfunc = kfunc
        self.nlfunc = None

        # Lower and upper bound arrays
        self.kparamsLB = None
        self.ynoiseLB = None
        self.nlparamsLB = None
        self.kparamsUB = None
        self.ynoiseUB = None
        self.nlparamsUB = None

        # Parameters
        self.obj = None
        self.m = None
        self.C = None
        self.Kchol = None

        # Hyperparameters
        self.kparams = None
        self.ynoise = None
        self.nlparams = None

        # Clear the data and hyperparameters
        self.__wipedata()

    def learn(self, x, y, kparams, ynoise, dobj=1e-5, dparams=1e-8, maxit=200,
              verbose=False):
        """ Learn method for learning the likelihood noise and kernel
            hyperparameters of the GP.

            Arguments:
                x: [DxN] array of N input samples with a dimensionality of D.
                y: N array of training outputs (dimensionality of 1)
                kparams: a tuple of initial values corresponding to the kernel
                    hyperparameters of the kernel function input to the
                    constructor.
                ynoise: a scalar initial value for the observation (y) noise.
                dobj: [optional] the convergence threshold for the objective
                    function (log-marginal likelihood) used by NLopt.
                dparams: [optional] the convergence threshold for the
                    hyperparameter values.
                maxit: [optional] maximum number of iterations for learning the
                    hyperparameters.
                verbose: [optional] whether or not to display current learning
                    progress to the terminal.

            Returns:
                The final objective function value (log marginal likelihood).
                Also internally all of the final parameters and hyperparameters
                are stored.

            Note:
                This stops learning as soon as one of the convergence criterion
                is met (objective, hyperparameters or iterations).

        """

        return self._learn(self.__gplearn, x, y, kparams, ynoise, None, dobj,
                           dparams, maxit, verbose)

    def predict(self, xs):
        """ Predict the outputs and thier variance, y* and V[y*], given new
            inputs, x*.

            Arguments:
                xs: [DxN] test points for prediction (x*)

            Returns:
                Ems: array of N predictions of m*
                Vms: array of N predictions of the variance of m*, V[m*].
        """

        # Check we have trained
        D, N = self._check_prediction_inputs(xs)

        # Pre-allocate
        Ems, Vms = np.zeros(N), np.zeros(N)

        # Evaluate test kernel vectors and do a stable inversion
        ks = self.kfunc(self.x, np.atleast_2d(xs), *self.kparams)
        Kinvks = cholsolve(jitchol(self.C), ks)

        for n, xn in enumerate(xs.T):

            # Evaluate the test kernel vectors
            kss = self.kfunc(np.atleast_2d(xn).T, np.atleast_2d(xn).T,
                             *self.kparams)

            # Predict the latent function
            Ems[n] = (Kinvks[:, n].T).dot(self.m)
            Vms[n] = kss - Kinvks[:, n].T.dot(ks[:, n])

        return Ems, Vms

    def learnLB(self, kparams=None, nlparams=None, ynoise=None):
        """ Set the lower bounds for the parameters to be learned.

            Arguments:
                kparams: a tuple of values corresponding to the kernel
                    hyperparameters of the kernel function input to the
                    constructor.
                nlparams: a tuple of values corresponding to the nonlinear
                    function parameters of the nonlinear function input to the
                    constructor (ignored for the basic BP).
                ynoise: a scalar value for the observation (y) noise.
        """

        self.kparamsLB = kparams
        self.nlparamsLB = nlparams
        self.ynoiseLB = ynoise

    def learnUB(self, kparams=None, nlparams=None, ynoise=None):
        """ Set the upper bounds for the parameters to be learned.

            Arguments:
                kparams: a tuple of values corresponding to the kernel
                    hyperparameters of the kernel function input to the
                    constructor.
                nlparams: a tuple of values corresponding to the nonlinear
                    function parameters of the nonlinear function input to the
                    constructor (ignored for the basic BP).
                ynoise: a scalar value for the observation (y) noise.
        """

        self.kparamsUB = kparams
        self.nlparamsUB = nlparams
        self.ynoiseUB = ynoise

    def _learn(self, learnfunc, x, y, kparams, ynoise, nlparams, dobj, dparams,
               maxit, verbose):
        """ Generic optimisation method for this and derived Gaussian Process
            algorithms.

            Essentially this manages the call to NLopt, establishes upper and
            lower bounds on the hyperparameters, and sets the internal
            parameters and hyperparameters to their learned values.

            Arguments:
                learnfunc: the learning function to call to actually learn the
                    *parameters* of the GP, i.e. the posterior mean and
                    covariance, it should have the following minimal form:


                    def learnfunc(y, K, delta=None, maxit=None, verbose=False):

                        ... do some calcs

                        return m, C, obj

                    where K is the prior [NxN] covariance matrix built using
                    the current estimates of the hyperparameters, m is the
                    posterior mean of the GP, C is the posterior covariance of
                    the GP and obj is the final value of the objective function
                    (log marginal likelihood or some proxy). Also:

                        delta: is the convergence threshold, and is passed
                            dobj/10
                        maxit: the maximum number of iterations to perform
                            (passed maxit from this function)
                        verbose: toggle verbose output, also routed from this
                            function.

                x: [DxN] array of N input samples with a dimensionality of D.
                y: N array of training outputs (dimensionality of 1)
                kparams: a tuple of initial values corresponding to the kernel
                    hyperparameters of the kernel function input to the
                    constructor.
                ynoise: a scalar initial value for the observation (y) noise.
                dobj: the convergence threshold for the objective function
                    (log-marginal likelihood) used by NLopt.
                dparams: the convergence threshold for the hyperparameter
                    values.
                maxit: maximum numbr of iterations for learning the
                    hyperparameters.
                verbose: whether or not to display current learning progress to
                    the terminal.

            Returns:
                The final objective function value (log marginal likelihood).
                Also internally all of the final parameters and hyperparameters
                are stored.

            Note:
                This stops learning as soon as one of the convergence criterion
                is met (objective, hyperparameters or iterations).
        """

        # Check arguments
        self.__wipedata()
        D, N = self.__check_training_inputs(x, y)

        # Check bounds with parameters to learn
        lbounds, ubounds = self.__checkbounds(kparams, nlparams, ynoise)

        # Make log-marginal-likelihood closure for optimisation
        def objective(params, grad):

            # Make sure grad is empty
            assert not grad, "Grad is not empty!"

            # Extract hyperparameters
            self.__extractparams(params, kparams, nlparams, ynoise)
            K = self.kfunc(self.x, self.x, *self.kparams)

            m, C, obj = learnfunc(y, K, delta=dobj/10, maxit=maxit,
                                  verbose=verbose)

            if obj > self.obj:
                self.m, self.C, self.obj = m, C, obj

            if verbose is True:
                print("\tObjective: {}, params: {}".format(obj, params))

            return obj

        # Get initial hyper-parameters
        params = self.__catparams(kparams, nlparams, ynoise)
        nparams = len(params)

        # Set up optimiser with objective function and bounds
        opt = nlopt.opt(nlopt.LN_BOBYQA, nparams)
        opt.set_max_objective(objective)
        opt.set_lower_bounds(lbounds)
        opt.set_upper_bounds(ubounds)
        opt.set_maxeval(maxit)
        opt.set_ftol_rel(dobj)
        opt.set_xtol_rel(dparams)

        # Run the optimisation
        params = opt.optimize(params)
        optfval = opt.last_optimize_result()

        if verbose is True:
            print("Optimiser finish criterion: {0}".format(optfval))

        # Store learned parameters (these have been over-written by nlopt)
        self.__extractparams(params, kparams, nlparams, ynoise)
        self.Kchol = jitchol(self.kfunc(self.x, self.x, *self.kparams))

        return self.obj

    def _quadpredict(self, xs):
        """ Prediction of m* and E[y*] using quadrature to evaluate E[y*]. This
            is primarily intended for the nonlinear GPs.

            Arguments:
                xs: [DxN] test points for prediction

            Returns:
                Eys: array of N predictions of E[y*]
                eEys: array of N errors on each E[y*] integral evaluation
                Ems: array of N predictions of m*
                Vms: array of N predictions of the variance of m*, V[m*].
        """

        # Check we have trained
        D, N = self._check_prediction_inputs(xs)

        # Pre-allocate
        Ems, Vms, Eys, eEys = np.zeros(N), np.zeros(N), np.zeros(N), \
            np.zeros(N)

        # Expected predicted target (to be integrated)
        def expecy(xsn, Emn, Vmn):

            gxs = self._passnlfunc(self.nlfunc, xsn)
            quad_msEf = (xsn - Emn)**2 / Vmn
            return gxs * np.exp(-0.5 * (quad_msEf + np.log(2 * np.pi * Vmn)))

        # Evaluate test kernel vectors and do a stable inversion
        ks = self.kfunc(self.x, np.atleast_2d(xs), *self.kparams)
        Kinvks = cholsolve(self.Kchol, ks)

        for n, xn in enumerate(xs.T):

            # Evaluate the test kernel vectors
            kss = self.kfunc(np.atleast_2d(xn).T, np.atleast_2d(xn).T,
                             *self.kparams)

            # Predict the latent function
            Ems[n] = (Kinvks[:, n].T).dot(self.m)
            Vms[n] = kss - Kinvks[:, n].T.dot(ks[:, n]
                                              - self.C.dot(Kinvks[:, n]))

            # Use Quadrature to get predicted target value
            st = 4 * np.sqrt(Vms[n])  # Evaluate the integral to 4 sig
            Eys[n], eEys[n] = spint.quad(expecy, a=Ems[n]-st, b=Ems[n]+st,
                                         args=(Ems[n], Vms[n]))

        return Eys, eEys, Ems, Vms

    def _check_prediction_inputs(self, xs):
        """ Check the prediction inputs for compatability.

            Arguments:
                xs: [DxN] test points for prediction
        """

        # Check we have trained
        if self.m is None:
            raise ValueError("This GP needs to be learned first!")

        D, N = (1, xs.shape[0]) if xs.ndim == 1 else xs.shape

        if D != self.x.shape[0]:
            raise ValueError("The test and training inputs are not the same"
                             " dimensionality!")

        return D, N

    def _passnlfunc(self, func, f):
        """ Pass points f though the nonlinear function func.

            This is a convenience method that also check whether or not
            nlparams exists, and calls func accordingly.

            Arguments:
                func: a function that can take a vector input, and give a
                    vector of the same length on the output.
                f: an array of N points to pass through func.

            Returns:
                an array of N points output from func(f).
        """

        return func(f) if self.nlparams is None else func(f, *self.nlparams)

    def __catparams(self, kparams, nlparams, ynoise):
        """ Concatenate the parameters for optimisation. """

        params = []
        if kparams is not None:
            params = kparams
        if nlparams is not None:
            params = np.hstack((params, nlparams))
        if ynoise is not None:
            params = np.hstack((params, [ynoise]))

        return params

    def __extractparams(self, params, kparams, nlparams=None, ynoise=None):
        """ Extract the model parameters after optimisation from a list. """

        self.kparams = params[0:len(kparams)]

        self.nlparams = None if not nlparams else \
            params[len(kparams):(len(kparams)+len(nlparams))]

        self.ynoise = None if not ynoise else params[-1]

    def __check_training_inputs(self, x, y):
        """ Check the training inputs for compatability. """

        # Check if there is already data
        if self.x is not None:
            raise ValueError("This GP already has data!")

        # Check arguments
        self.x = np.array(x[np.newaxis, :]) if x.ndim == 1 else np.array(x)
        (D, N) = self.x.shape

        if y.shape != (N,):
            self.x = None
            raise ValueError("x and y do not have the same number of points"
                             " or y is the wrong shape!")

        return D, N

    def __wipedata(self):
        """ Clear this object """

        # Hyper-parameters
        self.kparams = None
        self.ynoise = None

        # Functions and their parameters
        self.nlparams = None

        # Matrices to store for fast computations
        self.x = None
        self.Kchol = None
        self.m = None
        self.C = None
        self.obj = -np.inf

    def __checkbounds(self, kparams, nlparams, ynoise):
        """ Make sure the parameter bounds match the starting parameters. """

        if self.kparamsLB is not None:
            if len(kparams) != len(self.kparamsLB):
                raise ValueError("Lower bound kernel parameters size mismatch")
        else:
            self.kparamsLB = -np.inf * np.ones(len(kparams))

        if self.kparamsUB is not None:
            if len(kparams) != len(self.kparamsUB):
                raise ValueError("Upper bound kernel parameters size mismatch")
        else:
            self.kparamsUB = np.inf * np.ones(len(kparams))

        if self.nlparamsLB is not None:
            if len(nlparams) != len(self.nlparamsLB):
                raise ValueError("Lower bound function parameters mismatch")
        elif nlparams is not None:
            self.nlparamsLB = -np.inf * np.ones(len(nlparams))

        if self.nlparamsUB is not None:
            if len(nlparams) != len(self.nlparamsUB):
                raise ValueError("Upper bound function parameters mismatch")
        elif nlparams is not None:
            self.nlparamsUB = np.inf * np.ones(len(nlparams))

        if (ynoise is not None) and (self.ynoiseLB is None):
            self.ynoiseLB = -np.inf
        elif (ynoise is None) and (self.ynoiseLB is not None):
            raise ValueError("No noise parameter, but lower bound set!")

        if (ynoise is not None) and (self.ynoiseUB is None):
            self.ynoiseUB = np.inf
        elif (ynoise is None) and (self.ynoiseUB is not None):
            raise ValueError("No noise parameter, but upper bound set!")

        lbounds = self.__catparams(self.kparamsLB, self.nlparamsLB,
                                   self.ynoiseLB)
        ubounds = self.__catparams(self.kparamsUB, self.nlparamsUB,
                                   self.ynoiseUB)

        return lbounds, ubounds

    def __gplearn(self, y, K, delta=None, maxit=None, verbose=False):
        """ Parameter learning for a basic GP. Called by the _learn method. """

        N = y.shape[0]

        # Make posterior GP
        C = K + np.eye(N) * self.ynoise**2
        Cchol = jitchol(C)

        # Calculate the log-marginal-likelihood
        lml = -0.5 * (logdet(Cchol) + y.T.dot(cholsolve(Cchol, y)) + N *
                      np.log(2 * np.pi))

        return y, C, lml
