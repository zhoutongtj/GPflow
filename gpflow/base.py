import functools
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tabulate import tabulate
from tensorflow.python.ops import array_ops

from .util import default_float, get_str_tensor_value

DType = Union[np.dtype, tf.DType]
VariableData = Union[List, Tuple, np.ndarray, int, float]
Transform = tfp.bijectors.Bijector
Prior = tfp.distributions.Distribution

positive = tfp.bijectors.Softplus
triangular = tfp.bijectors.FillTriangular


def _IS_PARAMETER(o):
    return isinstance(o, Parameter)


def _IS_TRAINABLE_PARAMETER(o):
    return (_IS_PARAMETER(o) and o.trainable)


class Module(tf.Module):
    @property
    def parameters(self):
        return self._flatten(predicate=_IS_PARAMETER)

    @property
    def trainable_parameters(self):
        return self._flatten(predicate=_IS_TRAINABLE_PARAMETER)


class Parameter(tf.Module):
    def __init__(self,
                 value,
                 *,
                 transform: Optional[Transform] = None,
                 prior: Optional[Prior] = None,
                 trainable: bool = True,
                 dtype: DType = None,
                 name: str = None):
        """
        Unconstrained parameter representation.
        According to standart terminology `y` is always transformed representation or,
        in other words, it is constrained version of the parameter. Normally, it is hard
        to operate with unconstrained parameters. For e.g. `variance` cannot be negative,
        therefore we need positive constraint and it is natural to use constrained values.
        """
        super().__init__()

        value = _verified_value(value, dtype)
        if isinstance(value, tf.Variable):
            self._unconstrained = value
        else:
            value = _to_unconstrained(value, transform)
            self._unconstrained = tf.Variable(value,
                                              dtype=dtype,
                                              name=name,
                                              trainable=trainable)

        self.prior = prior
        self._transform = transform

    def log_prior(self):
        x = self.read_value()
        y = self._unconstrained
        dtype = x.dtype

        log_prob = tf.convert_to_tensor(0., dtype=dtype)
        log_det_jacobian = tf.convert_to_tensor(0., dtype=dtype)

        bijector = self.transform
        if self.prior is not None:
            log_prob = self.prior.log_prob(x)
        if self.transform is not None:
            log_det_jacobian = bijector.forward_log_det_jacobian(
                y, y.shape.ndims)
        return log_prob + log_det_jacobian

    @property
    def handle(self):
        return self._unconstrained.handle

    def value(self):
        return _to_constrained(self._unconstrained.value(), self.transform)

    def read_value(self):
        return _to_constrained(self._unconstrained.read_value(),
                               self.transform)

    @property
    def unconstrained_variable(self):
        return self._unconstrained

    @property
    def transform(self):
        return self._transform

    @property
    def trainable(self):
        return self._unconstrained.trainable

    @trainable.setter
    def trainable(self, flag: Union[bool, int]):
        self._unconstrained._trainable = bool(flag)

    @property
    def initial_value(self):
        return self._unconstrained.initial_value

    def assign(self, value, use_locking=False, name=None, read_value=True):
        # TODO(sergio.pasc): Find proper solution for casting / Discuss solution
        value = _verified_value(value, self.dtype)
        unconstrained_value = _to_unconstrained(value, self.transform)

        self._unconstrained.assign(unconstrained_value,
                                   read_value=read_value,
                                   use_locking=use_locking)

    @property
    def name(self):
        return self._unconstrained.name

    @property
    def initializer(self):
        return self._unconstrained.initializer

    @property
    def device(self):
        return self._unconstrained.device

    @property
    def dtype(self):
        return self._unconstrained.dtype

    @property
    def graph(self):
        return self._unconstrained.graph

    @property
    def op(self):
        return self._unconstrained.op

    @property
    def shape(self):
        if self.transform is not None:
            return self.transform.forward_event_shape(
                self._unconstrained.shape)
        return self._unconstrained.shape

    def numpy(self):
        return self.read_value().numpy()

    def get_shape(self):
        return self.shape

    def _should_act_as_resource_variable(self):
        pass

    def __repr__(self):
        return self.read_value().__repr__()

    def __ilshift__(self, value: VariableData) -> 'Parameter':
        self.assign(tf.cast(value, self.dtype))
        return self

    # Below
    # TensorFlow copy-paste code to make variable-like object to work

    @classmethod
    def _OverloadAllOperators(cls):  # pylint: disable=invalid-name
        """Register overloads for all operators."""
        for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
            cls._OverloadOperator(operator)
        # For slicing, bind getitem differently than a tensor (use SliceHelperVar
        # instead)
        # pylint: disable=protected-access
        setattr(cls, "__getitem__", array_ops._SliceHelperVar)

    @classmethod
    def _OverloadOperator(cls, operator):  # pylint: disable=invalid-name
        """Defer an operator overload to `ops.Tensor`.

        We pull the operator out of ops.Tensor dynamically to avoid ordering issues.

        Args:
            operator: string. The operator name.
        """
        tensor_oper = getattr(tf.Tensor, operator)

        def _run_op(a, *args, **kwargs):
            # pylint: disable=protected-access
            return tensor_oper(a.read_value(), *args, **kwargs)

        functools.update_wrapper(_run_op, tensor_oper)
        setattr(cls, operator, _run_op)

    # NOTE(mrry): This enables the Variable's overloaded "right" binary
    # operators to run when the left operand is an ndarray, because it
    # accords the Variable class higher priority than an ndarray, or a
    # numpy matrix.
    # TODO(mrry): Convert this to using numpy's __numpy_ufunc__
    # mechanism, which allows more control over how Variables interact
    # with ndarrays.
    __array_priority__ = 100


Parameter._OverloadAllOperators()
tf.register_tensor_conversion_function(
    Parameter, lambda x, *args, **kwds: x.read_value())


def _verified_value(value: VariableData,
                    dtype: Optional[DType] = None) -> np.ndarray:
    if isinstance(value, tf.Variable):
        return value
    if dtype is None:
        dtype = default_float()
    return tf.cast(value, dtype)


def _to_constrained(value: VariableData, transform: Transform) -> tf.Tensor:
    if transform is not None:
        return transform.forward(value)
    return value


def _to_unconstrained(value: VariableData, transform: Transform) -> tf.Tensor:
    if transform is not None:
        return transform.inverse(value)
    return value


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
    ] for path, variable in get_component_variables(module)]

    print(tabulate(column_values, headers=column_names, tablefmt=fmt))


def get_component_variables(module: tf.Module, prefix=None):
    prefix = module.__class__.__name__ if prefix is None else prefix
    var_list = []
    module_dict = vars(module)
    for key, submodule in module_dict.items():
        if key in tf.Module()._TF_MODULE_IGNORED_PROPERTIES:
            continue
        elif isinstance(submodule, Parameter) or isinstance(submodule, tf.Variable):
            var_list.append(('%s.%s' % (prefix, key), submodule))
        elif isinstance(submodule, tf.Module):
            submodule_var = get_component_variables(submodule, prefix='%s.%s' % (prefix, key))
            var_list.extend(submodule_var)
        elif isinstance(submodule, list):
            for idx, item in enumerate(submodule):
                item_name = item.__class__.__name__
                if key in ['_trainable_weights']:
                    continue
                elif isinstance(item, tf.Module):
                    submodule_var = get_component_variables(
                        item, prefix='%s.%s.%s_%i' % (prefix, key, item_name, idx)
                    )
                    var_list.extend(submodule_var)
                elif isinstance(item, Parameter) or isinstance(item, tf.Variable):
                    var_list.append(('%s.%s.%s_%i' % (prefix, key, item_name, idx), item))


    return var_list
