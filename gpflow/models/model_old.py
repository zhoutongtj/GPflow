# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import abc

import tensorflow as tf

from gpflow.mean_functions import Zero
from gpflow.models import BayesianModel
from gpflow.models.model import MeanAndVariance
from gpflow.util import default_float, default_jitter


class GPModelOLD(BayesianModel):
    """
    A base class for Gaussian process models, that is, those of the form

    .. math::
       :nowrap:

       \\begin{align}
       \\theta & \sim p(\\theta) \\\\
       f       & \sim \\mathcal{GP}(m(x), k(x, x'; \\theta)) \\\\
       f_i       & = f(x_i) \\\\
       y_i\,|\,f_i     & \sim p(y_i|f_i)
       \\end{align}

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.

    For handling another data (Xnew, Ynew), set the new value to self.X and self.Y

    >>> m.X = Xnew
    >>> m.Y = Ynew
    """

    def __init__(self,
                 X: object,
                 Y: object,
                 kernel: object,
                 likelihood: object,
                 mean_function: object = None,
                 num_latent: object = 1,
                 seed: object = None) -> object:
        super().__init__()
        self.X = X
        self.Y = Y
        self.num_latent = num_latent or Y.shape[1]
        #TODO(@awav): Why is this here when MeanFunction does not have a __len__ method
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        self.kernel = kernel
        self.likelihood = likelihood

    @abc.abstractmethod
    def predict_f(self, X: tf.Tensor, full=False, full_output_cov=False) -> MeanAndVariance:
        pass

    def predict_f_samples(self, X, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.predict_f(X, full=True)  # [P, N, N]
        jitter = tf.eye(tf.shape(mu)[0], dtype=default_float()) * default_jitter()
        samples = [None] * self.num_latent
        for i in range(self.num_latent):
            L = tf.linalg.cholesky(var[i, ...] + jitter)
            shape = tf.stack([L.shape[0], num_samples])
            V = tf.random.normal(shape, dtype=L.dtype)
            samples[i] = mu[:, i:(i+1)] + L @ V
        return tf.transpose(tf.stack(samples))

    def predict_y(self, X):
        """
        Compute the mean and variance of held-out data at the points X
        """
        f_mean, f_var = self.predict_f(X)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_log_density(self, X, Y):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        f_mean, f_var = self.predict_f(X)
        return self.likelihood.predict_density(f_mean, f_var, Y)
