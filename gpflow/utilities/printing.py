# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from functools import lru_cache

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from .. import Parameter


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
