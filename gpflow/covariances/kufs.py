import tensorflow as tf

from gpflow.util import Register
from ..features import InducingPoints, Multiscale
from ..kernels import Kernel, RBF
from .dispatch import Kuf_dispatcher


def Kuf(feature: InducingPoints, kernel: Kernel, Xnew):
    kuf_fn = Kuf_dispatcher.registered_fn(type(feature), type(kernel))
    return kuf_fn(feature, kernel, Xnew)


@Register(Kuf_dispatcher, InducingPoints, Kernel)
def _Kuf(feature: InducingPoints, kernel: Kernel, Xnew: tf.Tensor):
    return kernel(feature.Z, Xnew)


@Register(Kuf_dispatcher, Multiscale, RBF)
def _Kuf(feature: Multiscale, kernel: RBF, Xnew):
    Xnew, _ = kernel.slice(Xnew, None)
    Zmu, Zlen = kernel.slice(feature.Z, feature.scales)
    idlengthscale = kernel.lengthscale + Zlen
    d = feature._cust_square_dist(Xnew, Zmu, idlengthscale)
    lengthscale = tf.reduce_prod(kernel.lengthscale / idlengthscale, 1)
    lengthscale = tf.reshape(lengthscale, (1, -1))
    return tf.transpose(kernel.variance * tf.exp(-d / 2) * lengthscale)
