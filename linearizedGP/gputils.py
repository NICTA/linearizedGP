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

""" General utilities useful for Gaussian processes.

    Author:     Daniel Steinberg (daniel.steinberg@nicta.com.au)
    Institute:  NICTA
    Date:       17 Mar 2014

"""

import numpy as np
import scipy.linalg as la


def jitchol(A):
    """ Do cholesky decomposition with a bit of diagonal jitter if needs be.

        Aarguments:
            A: a [NxN] positive definite symmetric matrix to be decomposed as
                A = L.dot(L.T).

        Returns:
            A lower triangular matrix factor, L, also [NxN].
    """

    # Try the cholesky first
    try:
        cholA = la.cholesky(A, lower=True)
        return cholA
    except la.LinAlgError as e:
        pass

    # Now add jitter
    D = A.shape[0]
    jit = 1e-13
    cholA = None
    di = np.diag_indices(D)
    Amean = A.diagonal().mean()

    while jit < 1e-3:

        try:
            Ajit = A.copy()
            Ajit[di] += Amean * jit
            cholA = la.cholesky(Ajit, lower=True)
            break
        except la.LinAlgError as e:
            jit *= 10

    if cholA is None:
        raise la.LinAlgError("Too much jit! " + e.message)

    return cholA


def cholsolve(L, b):
    """ Solve the system of equations Ax = b with the cholesky A = L*L.T

        Arguments:
            L: A [NxN] lower triangular cholesky factor.
            b: A [NxD] matrix or N vector.

        Return:
            x: a [NxD] matrix or N vector solution.
    """

    return la.solve_triangular(L.T, la.solve_triangular(L, b, lower=True))


def k_fold_CV(X, Y, k=5):
    """ Generator to divide a dataset k non-overlapping folds.

        Author: Lachlan McCalman
        Modified: Daniel Steinberg

        Arguments:
            X: Input data [DxN] where D is the dimensionality, and N is the
                number of samples (X can also be a 1-d vector).
            Y: Output data vector of length N.
            k: [optional] the number of folds for testing and training.

        Returns (per call):
            Xr: [D x ((k-1) * N / k)] training input data
            Yr: [(k-1) * N / k] training output data
            Xs: [D x (N / k)] testing input data
            Ys: [N / k] testing output data

            All of these are randomly split (but non-overlapping per call)
    """

    X = np.atleast_2d(X)
    random_indices = np.random.permutation(X.shape[1])
    X = X[:, random_indices]
    Y = Y[random_indices]
    X_groups = np.array_split(X, k, axis=1)
    Y_groups = np.array_split(Y, k)

    for i in range(k):
        X_s = X_groups[i]
        Y_s = Y_groups[i]
        X_r = np.hstack(X_groups[0:i] + X_groups[i + 1:])
        Y_r = np.concatenate(Y_groups[0:i] + Y_groups[i + 1:])
        yield (X_r, Y_r, X_s, Y_s)


def k_fold_CV_ind(nsamples, k=5):
    """ Generator to return random test and training indeces for cross fold
        validation.

        Arguments:
            nsamples: the number of samples in the dataset
            k: [optional] the number of folds

        Returns:
            rind: training indices of length nsamples * (k-1)/k
            sind: testing indices of length nsamples * 1/k

            Each call to this generator returns a random but non-overlapping
            split of data.
    """

    pindeces = np.random.permutation(nsamples)
    pgroups = np.array_split(pindeces, k)

    for i in range(k):
        sind = pgroups[i]
        rind = np.concatenate(pgroups[0:i] + pgroups[i + 1:])
        yield (rind, sind)


def logdet(L, dochol=False):
    """ Compute the log determinant of a matrix.

        Arguments:
            L: The [NxN] cholesky factor of the matrix if dochol is False,
                otherwise the original [NxN] matrix if dochol is True.
            dochol: [optional] do a cholesky decomposition on the input matrix
                L if it is not already as cholesky factor.

        Returns:
            The log determinant (scalar)
    """

    if dochol is True:
        L = jitchol(L)

    return 2 * np.log(L.diagonal()).sum()
