import tensorflow as tf

from ..multidispatch import Dispatch

Kuu_dispatch = Dispatch('Kuu', [('feature', 0), ('kernel', 1)])
Kuf_dispatch = Dispatch('Kuf', [('feature', 0), ('kernel', 1)])


def Kuu(feature, kernel, jitter=0.0):
    kuu_fn = Kuu_dispatch.registered_function(type(feature), type(kernel))
    return kuu_fn(feature, kernel, jitter=jitter)


def Kuf(feature, kernel, Xnew: tf.Tensor):
    kuf_fn = Kuf_dispatch.registered_function(type(feature), type(kernel))
    return kuf_fn(feature, kernel, Xnew)
