from typing import Union

import pytest

from gpflow.kernels import Kernel, RBF, Matern12, Matern32
from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.mean_functions import MeanFunction, Linear, Identity, Constant
from gpflow.multidispatch import Dispatch

dispatch = Dispatch("test", [("a", 0), ("b", 2)])


class A(int):
    pass


class B:
    pass

# @dispatch.register(a=float, b=Union[B, float])  # noqa: F811
# @dispatch.exclusive_register(a=[A, B], b=int)
# def __func(a, c, b):
#     return None
#
# @dispatch  # noqa: F811
# def __func(a: int, c: object, b: Union[A, B]):
#     return a * b
#
# @dispatch.exclusive  # noqa: F811
# def __func(a: float, c: object, b: float):
#     return a * b
#
# @dispatch.register(a=[int, float], b=[A, B])  # noqa: F811
# def __func(a, c: object, b):
#     return a * b
#
#
#
# print(func(10, 0, 2.))
# print(func(10, 0, 2.))


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------

class Data:
    _storage_dict = {}
    _storage_dict[('A0', 'B1', 'C1')] = 2
    _storage_dict[('A1', 'B2', 'C0')] = 3
    _storage_dict[('A1', 'B0', 'C0')] = 1
    hierarchy_A = ['A0', 'A1']
    hierarchy_B = ['B0', 'B1', 'B2']
    hierarchy_C = ['C0', 'C1', 'C2']

# -------------------------------------------
# Test functions in utils
# -------------------------------------------

@pytest.mark.parametrize('type_a, type_b, type_c, dist, expected_min_dist', [
    ['A0', 'B0', 'C0', 0, 0],
    ['A1', 'B1', 'C0', 2, 1],
    ['A0', 'B2', 'C2', 4, 1],
])
def test_dispatch_search_for_candidate(type_a, type_b, type_c, dist, expected_min_dist):
    """
    Checks that _search_candidate finds the item in _storage_dict with the lowest distance to children
    nodes.
    """
    dispatch = Dispatch('test')
    dispatch._storage_dict = Data._storage_dict.copy()
    hierarchies = [Data.hierarchy_A, Data.hierarchy_B, Data.hierarchy_C]
    dispatch._storage_dict[(type_a, type_b, type_c)] = dist
    min_distance_to_children, _ = dispatch._search_for_candidate(hierarchies, 0, [])

    assert min_distance_to_children == expected_min_dist
    assert min_distance_to_children == min(dispatch._storage_dict.values())


@pytest.mark.parametrize('type_a, type_b, type_c', [
    [RBF, Gaussian, Constant],
    [Matern12, Gaussian, Identity],
    [Matern32, Gaussian, Identity],
])
def test_dispatch_registered_function_implemented(type_a, type_b, type_c):
    """
    Checks that the method registered_function returns the correct (registered) function for given
    argument types.
    """
    dispatch = Dispatch('test', [('kernel', 0), ('likelihood', 1), ('mean_function', 2)])

    @dispatch.register((RBF, Matern12), Gaussian, (Constant, Identity))
    def function_1(kernel, likelihood, mean_function):
        pass

    if type_a is RBF or type_a is Matern12:
        fn = dispatch.registered_function(type_a, type_b, type_c)
        assert fn is not None
    else:
        with pytest.raises(NotImplementedError):
            fn = dispatch.registered_function(type_a, type_b, type_c)


@pytest.mark.parametrize('type_a, type_b, type_c, expected_fn_name', [
    [RBF, Gaussian, Constant, 'function_1'],
    [RBF, Gaussian, Identity, 'function_1'],
    [Matern12, Gaussian, Identity, 'function_1'],
    [Matern32, Gaussian, Identity, 'base_function'],
])
def test_dispatch_registered_function(type_a, type_b, type_c, expected_fn_name):
    """
    Checks that the method registered_function returns the correct (registered) function for given
    argument types.
    """
    dispatch = Dispatch('test', [('kernel', 0), ('likelihood', 1), ('mean_function', 2)])

    @dispatch.register((RBF, Matern12), Gaussian, (Constant, Identity))
    def function_1(kernel, likelihood, mean_function):
        pass

    @dispatch.register(Kernel, Likelihood, MeanFunction)
    def base_function(kernel, likelihood, mean_function):
        pass

    fn = dispatch.registered_function(type_a, type_b, type_c)
    assert fn.__name__ == expected_fn_name


def test_register_adding_single_type_functions():
    dispatch = Dispatch('test', [('kernel', 0), ('likelihood', 1), ('mean_function', 2)])

    @dispatch.register(RBF, Gaussian, Constant)
    def function_1(kernel, likelihood, mean_function):
        pass

    @dispatch
    def function_2(kernel: RBF, likelihood: Gaussian, mean_function: Identity):
        pass

    def function_3(kernel, likelihood, mean_function):
        pass

    dispatch.register(Matern12, Likelihood, Identity)(function_3)

    assert len(dispatch._storage_dict) == 3
    assert dispatch._storage_dict.get((RBF, Gaussian, Constant)) is function_1
    assert dispatch._storage_dict.get((RBF, Gaussian, Identity)) is function_2
    assert dispatch._storage_dict.get((Matern12, Likelihood, Identity)) is function_3

    with pytest.raises(ValueError):
        @dispatch
        def repeat_function_2(kernel: RBF, likelihood: Gaussian, mean_function: Identity):
            pass

    with pytest.raises(TypeError):
        def called_function():
            pass

        dispatch.register(kernel=RBF, likelihood=Gaussian, mean_function=Identity)(
            called_function())


def test_register_breaksdown_multiple_type_functions():
    dispatch = Dispatch('test', [('kernel', 0), ('likelihood', 1), ('mean_function', 2)])

    @dispatch.register(kernel=(RBF, Matern12),
                       likelihood=Gaussian,
                       mean_function=(Constant, Identity))
    def function_1(kernel, likelihood, mean_function):
        pass

    @dispatch
    def function_2(kernel: RBF, likelihood: Gaussian, mean_function: Identity):
        pass

    assert len(dispatch._storage_dict) == 5
    assert dispatch._storage_dict.get((RBF, Gaussian, Constant)) is function_1
    assert dispatch._storage_dict.get((RBF, Gaussian, Identity)) is function_1
    assert dispatch._storage_dict.get((Matern12, Gaussian, Constant)) is function_1
    assert dispatch._storage_dict.get((Matern12, Gaussian, Identity)) is function_1
    assert dispatch._storage_dict.get((RBF, Gaussian, Linear)) is function_2

    # with pytest.raises(ValueError):
    @dispatch
    def repeat_function_2(kernel: RBF, likelihood: Gaussian, mean_function: Identity):
        pass
