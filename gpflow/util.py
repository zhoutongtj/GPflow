import copy
import logging
import re
from functools import lru_cache
from typing import Callable, List, Union

import numpy as np
import tensorflow as tf
from tabulate import tabulate

NoneType = type(None)


def create_logger(name=None):
    return logging.getLogger('Temporary Logger Solution')


def default_jitter_eye(num_rows: int, num_columns: int = None, value: float = None) -> float:
    value = default_jitter() if value is None else value
    num_rows = int(num_rows)
    num_columns = int(num_columns) if num_columns is not None else num_columns
    return tf.eye(num_rows, num_columns=num_columns, dtype=default_float()) * value


def default_jitter() -> float:
    return 1e-6


def default_float() -> float:
    return np.float64


def default_int() -> int:
    return np.int32


def leading_transpose(tensor: tf.Tensor, perm: List[Union[int, type(...)]], leading_dim: int = 0) -> tf.Tensor:
    """
    Transposes tensors with leading dimensions. Leading dimensions in
    permutation list represented via ellipsis `...`.
    When leading dimensions are found, `transpose` method
    considers them as a single grouped element indexed by 0 in `perm` list. So, passing
    `perm=[-2, ..., -1]`, you assume that your input tensor has [..., A, B] shape,
    and you want to move leading dims between A and B dimensions.
    Dimension indices in permutation list can be negative or positive. Valid positive
    indices start from 1 up to the tensor rank, viewing leading dimensions `...` as zero
    index.
    Example:
        a = tf.random.normal((1, 2, 3, 4, 5, 6))  # [..., A, B, C],
                                                  # where A is 1st element,
                                                  # B is 2nd element and
                                                  # C is 3rd element in
                                                  # permutation list,
                                                  # leading dimentions are [1, 2, 3]
                                                  # which are 0th element in permutation
                                                  # list
        b = leading_transpose(a, [3, -3, ..., -2])  # [C, A, ..., B]
        sess.run(b).shape
        output> (6, 4, 1, 2, 3, 5)
    :param tensor: TensorFlow tensor.
    :param perm: List of permutation indices.
    :returns: TensorFlow tensor.
    :raises: ValueError when `...` cannot be found.
    """
    perm = copy.copy(perm)
    idx = perm.index(...)
    perm[idx] = leading_dim

    rank = tf.rank(tensor)
    perm_tf = perm % rank

    leading_dims = tf.range(rank - len(perm) + 1)
    perm = tf.concat([perm_tf[:idx], leading_dims, perm_tf[idx + 1:]], 0)
    return tf.transpose(tensor, perm)


def set_trainable(model: tf.Module, flag: bool = False):
    for variable in model.trainable_variables:
        variable._trainable = flag


def training_loop(closure: Callable[..., tf.Tensor],
                  optimizer=tf.optimizers.Adam(),
                  var_list: List[tf.Variable] = None,
                  jit=True,
                  maxiter=1e3):
    """
    Simple generic training loop. At each iteration uses a GradientTape to compute
    the gradients of a loss function with respect to a set of variables.

    :param closure: Callable that constructs a loss function based on data and model being trained
    :param optimizer: tf.optimizers or tf.keras.optimizers that updates variables by applying the
        corresponding loss gradients. Adam is a default optimizer with default settings.
    :param var_list: List of model variables to be learnt during training
    :param maxiter: Maximum number of
    :return:
    """

    def optimization_step():
        with tf.GradientTape() as tape:
            tape.watch(var_list)
            loss = closure()
            grads = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

    if jit:
        optimization_step = tf.function(optimization_step)

    for _ in range(int(maxiter)):
        optimization_step()


def broadcasting_elementwise(op, a, b):
    """
    Apply binary operation `op` to every pair in tensors `a` and `b`.

    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    :param a: tf.Tensor, shape [n_1, ..., n_a]
    :param b: tf.Tensor, shape [m_1, ..., m_b]
    :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    """
    flatres = op(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
    return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))


def print_summary(module: tf.Module, fmt: str = None):
    """
    Prints a summary of the parameters and variables contained in a tf.Module and its components.
    """
    fmt = fmt if fmt is not None else "simple"
    column_names = ['name', 'class', 'transform', 'trainable', 'shape', 'dtype', 'value']

    def get_name(v):
        return v.__class__.__name__

    def get_transform(v):
        if hasattr(v, 'transform') and v.transform is not None:
            return v.transform.__class__.__name__
        return None

    column_values = [[
        path,
        get_name(variable),
        get_transform(variable),
        variable.trainable,
        variable.shape,
        variable.dtype.name,
        get_str_tensor_value(variable.numpy())
    ] for path, variable in module.parameter_list()]

    print(tabulate(column_values, headers=column_names, tablefmt=fmt))


@lru_cache()
def _first_three_elements_regexp():
    num_re = r"[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?"
    pat_re = rf"^(?:(\[+)\s*)?({num_re})(?:\s+({num_re})(?:\s+({num_re}))?)?.*?"
    return re.compile(pat_re)


def get_str_tensor_value(value: np.ndarray):
    value_str = str(value)
    if value.size <= 3:
        return value_str

    max_chars = 500
    value_str = value_str[:max_chars]
    regexp = _first_three_elements_regexp()
    match = regexp.match(value_str)
    assert match is not None
    brackets, elem1, elem2, elem3 = match.groups()

    out = f"{elem1}"
    if elem2 is not None:
        out = f"{out}{f', {elem2}'}"
        if elem3 is not None:
            out = f"{out}{f', {elem3}'}"
    if brackets is not None:
        out = f"{brackets}{out}..."

    return out
