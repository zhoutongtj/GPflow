# pylint: disable=wildcard-import
from typing import Union
import tensorflow as tf

from .scipy import Scipy

all_optimizer_classes = {
    'scipy': Scipy
}


def get_optimizer(optimizer: Union[str, tf.optimizers.Optimizer]):
    all_optimizer_classes = {
        'scipy': Scipy
    }
    if isinstance(optimizer, str):
        if optimizer in all_optimizer_classes.keys():
            return all_optimizer_classes.get(optimizer)()
        else:
            return tf.keras.optimizers.get(optimizer)
    elif isinstance(optimizer, tf.optimizers.Optimizer):
        return optimizer
    else:
        return tf.optimizers.Adam()
