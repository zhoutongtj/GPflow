import copy
import logging
import itertools
from typing import Callable, List, Tuple, Union, Type, TypeVar

import numpy as np
import tensorflow as tf
from tensorflow.python.util import tf_inspect


__all__ = [
    'Dispatch',
    'DispatchArgspec',
    'DispatchArgspecs',
]


DispatchArgspec = TypeVar('DispatchArgspec', str, Tuple[str, int])
DispatchArgspecs = List[DispatchArgspec]


class Dispatch:
    def __init__(self, name: str, argspecs: DispatchArgspecs):
        self._name = name
        self._storage_dict = dict()
        self._dispatch_specs = valid_dispatch_argspecs(argspecs)

    @property
    def name(self):
        return self._name

    def registered_function(self, *types):
        key = tuple(types)
        if key not in self._storage_dict:
            raise ValueError(f"Dispatcher does not have registered function for '{key}'")
        return self._storage_dict[key]

    def __call__(self, func: Callable):
        self._register_using_func_annotations(func, use_subclasses=True)
        return func

    def exclusive(self, func: Callable):
        self._register_using_func_annotations(func, use_subclasses=False)
        return func

    def exclusive_register(self, **kwargs):
        def register(func: Callable):
            self._register_using_specified_types(kwargs, func, use_subclasses=False)
            return func
        return register

    def register(self, **kwargs):
        def register(func: Callable):
            self._register_using_specified_types(kwargs, func, use_subclasses=True)
            return func
        return register

    def _extract_types(self, handle_cb: Callable):
        types = []
        for spec in self._dispatch_specs:
            annotation = handle_cb(spec)
            if type(annotation) == type(Union):
                annotation = annotation.__args__
            if isinstance(annotation, TypeVar):
                annotation = annotation.__constraints__
            if not isinstance(annotation, (list, tuple)):
                annotation = [annotation]
            types.append(list(annotation))
        return types

    def _extract_types_by_dispatch_args(self, kwargs):
        dispatch_argnames = set(dict(self._dispatch_specs).keys())
        diff = dispatch_argnames.symmetric_difference(kwargs.keys())
        if diff:
            raise ValueError(f"Dispatch register kwargs do not match specification. "
                             f"Expected: {dispatch_argnames}, gotten: {set(kwargs.keys())}")

        def annotation_cb(spec):
            argname, _ind = spec
            return kwargs[argname]

        return self._extract_types(annotation_cb)

    def _extract_types_by_func_annotations(self, func: Callable, include_parents=True):
        argspec = tf_inspect.getfullargspec(func)

        def annotation_cb(spec):
            argname = argname_from_argspec(argspec, spec)
            return annotation_from_argspec(argspec, argname)

        return self._extract_types(annotation_cb)

    def _register_function(self, argtypes, func: Callable, use_subclasses: bool):
        if use_subclasses:
            argtypes = list(extend_with_subclasses(argtypes))
        argtype_keys = cross_product(argtypes)
        for argkey in argtype_keys:
            if argkey in self._storage_dict:
                # Print that already used by other combination
                continue
            self._storage_dict[argkey] = func

    def _register_using_specified_types(self, kwargs, func: Callable, use_subclasses=True):
        argtypes = self._extract_types_by_dispatch_args(kwargs)
        self._register_function(argtypes, func, use_subclasses)

    def _register_using_func_annotations(self, func: Callable, use_subclasses=True):
        argtypes = self._extract_types_by_func_annotations(func)
        self._register_function(argtypes, func, use_subclasses)


# Dispatcher helpers


def valid_dispatch_argspecs(specs: DispatchArgspecs):
    valid_specs = []
    for spec in specs:
        if isinstance(spec, str):
            valid_specs.append((spec, None))
        elif isinstance(spec, (list, tuple)) and \
                isinstance(spec[0], str) and \
                isinstance(spec[1], int) and \
                len(spec) == 2:
            valid_specs.append(tuple(spec))
        else:
            raise TypeError(f"Not supported dispatcher specification: '{spec}'")
    return valid_specs


def argname_from_argspec(argspec, dispatcher_spec: DispatchArgspec):
    argname, ind = dispatcher_spec
    if argname not in argspec.args:
        raise ValueError(f"Dispatcher did not find the argument '{argname}' in the function.")
    if ind is not None:
        args_length = len(argspec.args)
        if ind >= args_length:
            raise ValueError(f"Requested argument index '{ind}', "
                             f"but the function has {args_length} arguments")
        name = argspec.args[ind]
        if name != argname:
            raise ValueError(f"According to dispatcher specification"
                             f"function positioned argument at {ind} is expected "
                             f"to be '{argname}', but it has '{name}' name")
    return argname


def annotation_from_argspec(argspec, name: str):
    if name not in argspec.annotations:
        raise ValueError(f"Function doesn't have an annotation for dispatching argument '{name}'")
    return argspec.annotations[name]


def cross_product(argtypes):
    combination = set(itertools.product(*argtypes))
    return combination


def extend_with_subclasses(argtypes: List[Union[Type, List[Type]]]):
    for i, argtype in enumerate(argtypes):
        if not isinstance(argtype, (list, tuple)):
            types_to_expand = [argtype]
        elif type(argtype) == type(Union):
            types_to_expand = argtype.__args__
        elif isinstance(argtype, (list, tuple)):
            types_to_expand = argtype

        argtype_with_parents = []
        for t in types_to_expand:
            subclasses = list(tf_inspect.getmro(t))
            if object in subclasses:  # Do not include "object" automatically
                subclasses.remove(object)
            argtype_with_parents.extend(subclasses)

        yield argtype_with_parents
