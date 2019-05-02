import copy
import logging
from typing import Callable, List, Union

import numpy as np
import tensorflow as tf
import typing
from tensorflow.python.util import tf_inspect

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


def leading_transpose(tensor: tf.Tensor, perm: List[Union[int, type(...)]],
                      leading_dim: int = 0) -> tf.Tensor:
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
    corresponding loss gradients
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

class Dispatcher:
    def __init__(self, name: str):
        self.name = name
        self.REF_DICT = {}

    def __repr__(self):
        return self.name

    def registered_fn(self, *types):
        """Gets the function registered for chosen classes."""
        hierarchies = [tf_inspect.getmro(type_arg) for type_arg in types]
        fn, _ = self._search_for_candidate(hierarchies, 0, [])
        return fn

    def register(self, *types):
        register_object = Register(self, *types)

        def _(func):
            register_object(func)
            return func

        return _

    def _search_for_candidate(self, hierarchies, parent_dist, parent_key, fn=None, dist=None):
        """Recursive function that finds and selects candidate functions in REF_DICT"""
        for mro_to_0, parent_0 in enumerate(hierarchies[0]):
            candidate_dist = parent_dist + mro_to_0
            candidate_key = parent_key + [parent_0]
            if len(hierarchies) > 1:
                fn, dist = self._search_for_candidate(hierarchies[1:], candidate_dist,
                                                      candidate_key, fn, dist)
            else:  # produce candidate
                candidate_fn = self.REF_DICT.get(tuple(candidate_key), None)
                candidate_exists_and_is_better = candidate_fn and candidate_dist < dist
                if not fn or candidate_exists_and_is_better:
                    dist = candidate_dist
                    fn = candidate_fn
        return fn, dist


class Register:
    def __init__(self, dispatch: Dispatcher, *types):
        self._key_list = self._breakdown_types(types)
        self._ref_dict = dispatch.REF_DICT
        self.name = dispatch.name

    def __call__(self, fn):
        """Perform the Multioutput Conditional registration.

        Args:
          fn: The function to use for the KL divergence.

        Returns:
          fn

        Raises:
          TypeError: if fn is not a callable.
          ValueError: if a function has already been registered for the given argument classes.
        """
        if not callable(fn):
            raise TypeError("fn must be callable, received: %s" % fn)
        for key in self._key_list:
            key = tuple(key)
            if key in self._ref_dict:
                raise ValueError("%s(%s, %s) has already been registered to: %s"
                                 % (self.name, key[0].__name__, key[1].__name__,
                                    self._ref_dict[key]))
            self._ref_dict[key] = fn
        return fn

    @staticmethod
    def _breakdown_types(types) -> List[List]:
        key_list = [[]]
        for x in types:
            if isinstance(x, tuple):
                key_list = [key + [y] for key in key_list for y in x]
            else:
                key_list = [key + [x] for key in key_list]
        return key_list

if __name__ == '__main__':
    # conditional_dispatcher = Dispatcher('conditional')
    #
    # @conditional_dispatcher.register(int, str, int)
    # def foo(a, b, c):
    #     print(a, b, c)
    #
    #
    # @conditional_dispatcher.register(str, int, str)
    # def foo2(a, b, c):
    #     pass
    #
    #
    # def foot(a, b, c):
    #     foo_fn = conditional_dispatcher.registered_fn(type(a), type(b), type(c))
    #     return foo_fn(a, b, c)
    #
    #
    # print(conditional_dispatcher.REF_DICT)
    # # foot('1',1,'a')
    # foot(0, '1', '1')

    #
    # REF_DICT = {}
    # REF_DICT[('A0', 'B1', 'C1')] = 'soln_first'
    # REF_DICT[('A1', 'B2', 'C1')] = 'wrong soln'
    # REF_DICT[('A1', 'B0', 'C0')] = 'soln_second'
    #
    # hierarchy_A = ['A0', 'A1']
    # hierarchy_B = ['B0', 'B1', 'B2']
    # hierarchy_C = ['C0', 'C1', 'C2']
    #
    #
    # def search_for_candidate(hierarchies, parent_dist, parent_key, fn=None, dist=None):
    #     for mro_to_0, parent_0 in enumerate(hierarchies[0]):
    #         candidate_dist = parent_dist + mro_to_0
    #         candidate_key = parent_key + [parent_0]
    #         if len(hierarchies) > 1:
    #             fn, dist = search_for_candidate(hierarchies[1:], candidate_dist,
    #                                             candidate_key, fn, dist)
    #         else:
    #             # produce candidate
    #             candidate_fn = REF_DICT.get(tuple(candidate_key), None)
    #             if candidate_fn is not None:
    #                 print(candidate_dist, candidate_fn, dist)
    #                 print(not fn)
    #                 print((candidate_fn and candidate_dist < dist))
    #             if not fn or (candidate_fn and candidate_dist < dist):
    #                 dist = candidate_dist
    #                 fn = candidate_fn
    #     return fn, dist
    #
    # fn = search_for_candidate([hierarchy_A, hierarchy_B, hierarchy_C], 0, [])
    # print(fn)
    a = list(('a', 'b', ('c', 'd'), ('e', 'f')))
    key_list = [[]]
    base = [[]] * len(a)
    for idx, x in enumerate(a):
        if isinstance(x, tuple):
            key_list = [key + [j] for key in key_list for j in x]
        else:
            key_list = [key+[x] for key in key_list]
    print(key_list)


