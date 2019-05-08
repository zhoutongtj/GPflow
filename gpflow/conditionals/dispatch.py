import tensorflow as tf
from ..multidispatch import Dispatch

conditional_dispatch = Dispatch('Conditional', [('feature', 1), ('kernel', 2)])
sample_conditional_dispatch = Dispatch('SampleConditional', [('feature', 1), ('kernel', 2)])


def conditional(Xnew: tf.Tensor,
                feature,
                kernel,
                function: tf.Tensor,
                full_cov=False,
                full_output_cov=False,
                q_sqrt=None,
                white=False):
    cb = conditional_dispatch.registered_function(type(feature), type(kernel))
    return cb(Xnew, feature, kernel, function, full_cov=full_cov, full_output_cov=full_output_cov,
              q_sqrt=q_sqrt, white=white)


def sample_conditional(Xnew: tf.Tensor,
                       feature,
                       kernel,
                       function: tf.Tensor,
                       full_cov=False,
                       full_output_cov=False,
                       q_sqrt=None,
                       white=False,
                       num_samples=None):
    cb = sample_conditional_dispatch.registered_function(type(feature), type(kernel))
    return cb(Xnew, feature, kernel, function, full_cov=full_cov, full_output_cov=full_output_cov,
              q_sqrt=q_sqrt, white=white, num_samples=num_samples)
