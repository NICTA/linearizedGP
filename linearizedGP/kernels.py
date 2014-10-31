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

""" Some generic kernel functions for this GP library.

    Author:     Daniel Steinberg (daniel.steinberg@nicta.com.au)
    Institute:  NICTA
    Date:       17 Mar 2014

"""

import numpy as np
from scipy.spatial.distance import cdist


# Kernel Functions -------------------------------------------------------

def kern_se(x, xs, sigma, length):
    """ Square Exponential Kernel.

        Arguments:
            x:  [DxN] array of D-dimensional input points
            xs: [DxN'] array of input points to evaluate w.r.t. x
            sigma: scalar amplitude kernel parameter
            length: scalar length scale kernel parameter

        Returns:
            [NxN'] matrix of similarities as evaluated by the kernel function
    """

    return sigma**2 * np.exp(- dist(x, xs) / (2 * length**2))


def kern_selog(x, xs, lsigma, llength):
    """ Square Exponential Kernel with parameters in log space.

        Arguments:
            x:  [DxN] array of D-dimensional input points
            xs: [DxN'] array of input points to evaluate w.r.t. x
            lsigma: log-scalar amplitude kernel parameter
            llength: log-scalar length scale kernel parameter

        Returns:
            [NxN'] matrix of similarities as evaluated by the kernel function
    """

    return np.exp(2*lsigma - dist(x, xs)/2 * np.exp(-2 * llength))


def kern_m32(x, xs, sigma, length):
    """ The Matern 3/2 Kernel.

        Arguments:
            x:  [DxN] array of D-dimensional input points
            xs: [DxN'] array of input points to evaluate w.r.t. x
            sigma: scalar amplitude kernel parameter
            length: scalar length scale kernel parameter

        Returns:
            [NxN'] matrix of similarities as evaluated by the kernel function
    """

    d = 3 * dist(x, xs)
    fd = np.sqrt(d) / length
    return sigma**2 * (1 + fd) * np.exp(-fd)


def kern_m52(x, xs, sigma, length):
    """ The Matern 5/2 Kernel.

        Arguments:
            x:  [DxN] array of D-dimensional input points
            xs: [DxN'] array of input points to evaluate w.r.t. x
            sigma: scalar amplitude kernel parameter
            length: scalar length scale kernel parameter

        Returns:
            [NxN'] matrix of similarities as evaluated by the kernel function
    """

    d = 5 * dist(x, xs)
    fd = np.sqrt(d) / length
    fd2 = d / length**2
    return sigma**2 * (1 + fd + fd2 / 3) * np.exp(-fd)


def dist(x, xs):
    """ Evaluate the square distance between all instances of x and xs.

        This just wraps scipy.spatial.distance.cdist()
    """

    return cdist(x.T, xs.T, 'sqeuclidean')
