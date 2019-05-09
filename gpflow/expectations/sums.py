import itertools
from functools import reduce
from typing import Union

import tensorflow as tf

from .dispatch import expectation_dispatch, expectation
from .. import kernels
from .. import mean_functions as mfn
from ..features import InducingPoints
from ..probability_distributions import (DiagonalGaussian, Gaussian,
                                         MarkovGaussian)
from ..util import NoneType


@expectation_dispatch
def _E(p: Gaussian, kernel: kernels.Sum, _: NoneType, __: NoneType, ___: NoneType, nghp=None):
    """
    Compute the expectation:
    <\Sum_i diag(Ki_{X, X})>_p(X)
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: N
    """
    exps = [expectation(p, k, nghp=nghp) for k in kernel.kernels]
    return reduce(tf.add, exps)


@expectation_dispatch
def _E(p: Gaussian, kernel: kernels.Sum, feature: InducingPoints,
       __: NoneType, ___: NoneType, nghp=None):
    """
    Compute the expectation:
    <\Sum_i Ki_{X, Z}>_p(X)
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: NxM
    """
    exps = [expectation(p, (k, feature), nghp=nghp) for k in kernel.kernels]
    return reduce(tf.add, exps)


@expectation_dispatch
def _E(p: Gaussian, mean: Union[mfn.Linear, mfn.Identity, mfn.Constant], _: NoneType,
       kernel:  kernels.Sum, feature: InducingPoints, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m(x_n)^T (\Sum_i Ki_{x_n, Z})>_p(x_n)
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: NxQxM
    """
    exps = [expectation(p, mean, (k, feature), nghp=nghp) for k in kernel.kernels]
    return reduce(tf.add, exps)


@expectation_dispatch
def _E(p: MarkovGaussian, mean: mfn.Identity, _: NoneType,
       kernel:  kernels.Sum, feature: InducingPoints, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <x_{n+1} (\Sum_i Ki_{x_n, Z})>_p(x_{n:n+1})
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: NxDxM
    """
    exps = [expectation(p, mean, (k, feature), nghp=nghp) for k in kernel.kernels]
    return reduce(tf.add, exps)


@expectation_dispatch((Gaussian, DiagonalGaussian), kernels.Sum, InducingPoints,
                                 kernels.Sum, InducingPoints)
def _E(p: Union[Gaussian, DiagonalGaussian], kern1:  kernels.Sum, feat1: InducingPoints,
       kern2:  kernels.Sum, feat2: InducingPoints,  nghp=None):
    """
    Compute the expectation:
    expectation[n] = <(\Sum_i K1_i_{Z1, x_n}) (\Sum_j K2_j_{x_n, Z2})>_p(x_n)
        - \Sum_i K1_i_{.,.}, \Sum_j K2_j_{.,.} :: Sum kernels

    :return: NxM1xM2
    """
    crossexps = []

    if kern1 == kern2 and feat1 == feat2:  # avoid duplicate computation by using transposes
        for i, k1 in enumerate(kern1.kernels):
            crossexps.append(expectation(p, (k1, feat1), (k1, feat1), nghp=nghp))

            for k2 in kern1.kernels[:i]:
                eKK = expectation(p, (k1, feat1), (k2, feat2), nghp=nghp)
                eKK += tf.linalg.adjoint(eKK)
                crossexps.append(eKK)
    else:
        for k1, k2 in itertools.product(kern1.kernels, kern2.kernels):
            crossexps.append(expectation(p, (k1, feat1), (k2, feat2), nghp=nghp))

    return reduce(tf.add, crossexps)
