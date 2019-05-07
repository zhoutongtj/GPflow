from typing import TypeVar

import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

Tensor = TypeVar('Tensor', tf.Tensor, tf.Variable, EagerTensor)
