import itertools
from typing import Callable, List, Tuple, Type, TypeVar, Union

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
        self._call_cache_dict = dict()
        self._storage_dict = dict()
        self._dispatch_specs = valid_dispatch_argspecs(argspecs)

    @property
    def name(self):
        return self._name

    def registered_function(self, *types):
        key = tuple(types)
        func_match = self._storage_dict.get(key, None)
        if func_match is not None:
            return func_match
        func_match = self._call_cache_dict.get(key, None)
        if func_match is None:
            func_match, _ = self._find_match_in_ancestors(key)
            if func_match is None:
                raise ValueError(f"Dispatcher does not have registered function for '{key}'")
            self._call_cache_dict[key] = func_match
        return func_match

    def __call__(self, func: Callable):
        self._register_using_func_annotations(func, use_ancestors=False)
        return func

    def register(self, **kwargs):
        def register(func: Callable):
            self._register_using_specified_types(kwargs, func, use_ancestors=False)
            return func

        return register

    def cross_product(self, func: Callable):
        self._register_using_func_annotations(func, use_ancestors=True)
        return func

    def cross_product_register(self, **kwargs):
        def register(func: Callable):
            self._register_using_specified_types(kwargs, func, use_ancestors=True)
            return func

        return register

    def _find_match_in_ancestors(self, key):
        key_with_ancestors = extend_with_ancestors(key)
        selected_types = cross_product(key_with_ancestors).intersection(set(self._storage_dict))
        sorted_selected_types = sorted(selected_types,
                                       key=lambda x: ranking_criteria(x, key_with_ancestors))
        selected_key = tuple(sorted_selected_types[0])
        return self._storage_dict[selected_key], selected_key

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

    def _register_function(self, argtypes, func: Callable, use_ancestors: bool):
        if use_ancestors:
            argtypes = extend_with_ancestors(argtypes)
        argtype_keys = cross_product(argtypes)
        for argkey in argtype_keys:
            if argkey in self._storage_dict:
                raise ValueError("%s with argument types %s has already been registered to: %s"
                                 % (self.name, (item.__name__ for item in argkey),
                                    self._storage_dict[argkey]))
            self._storage_dict[argkey] = func

    def _register_using_specified_types(self, kwargs, func: Callable, use_ancestors=True):
        argtypes = self._extract_types_by_dispatch_args(kwargs)
        self._register_function(argtypes, func, use_ancestors)

    def _register_using_func_annotations(self, func: Callable, use_ancestors=True):
        argtypes = self._extract_types_by_func_annotations(func)
        self._register_function(argtypes, func, use_ancestors)


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
        elif isinstance(spec, (list, tuple)) and \
                isinstance(spec[0], list) and \
                isinstance(spec[1], int) and \
                len(spec) == 2:
            valid_specs.extend([(arg_name, spec[1]) for arg_name in spec[0]])
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


def extend_with_ancestors(argtypes: List[Union[Type, List[Type]]]):
    argtypes_with_ancestors = []
    for i, argtype in enumerate(argtypes):
        if not isinstance(argtype, (list, tuple)):
            types_to_expand = [argtype]
        elif type(argtype) == type(Union):
            types_to_expand = argtype.__args__
        elif isinstance(argtype, (list, tuple)):
            types_to_expand = argtype

        argtype_with_parents = []
        for t in types_to_expand:
            ancestors = list(tf_inspect.getmro(t))
            if object in ancestors:  # Do not include "object" automatically
                ancestors.remove(object)
            argtype_with_parents.extend(ancestors)

        argtypes_with_ancestors.append(argtype_with_parents)
    return argtypes_with_ancestors


def ranking_criteria(types, hierarchies):
    return sum(map(lambda x: x[1].index(x[0]), zip(types, hierarchies)))
