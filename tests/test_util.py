# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import pytest

from gpflow.kernels import Kernel, Stationary, RBF, Matern12, Matern32
from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.mean_functions import MeanFunction, Linear, Identity, Constant
from gpflow.util import Dispatcher, Register


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------

class Data:
    REF_DICT = {}
    REF_DICT[('A0', 'B1', 'C1')] = 2
    REF_DICT[('A1', 'B2', 'C0')] = 3
    REF_DICT[('A1', 'B0', 'C0')] = 1
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
def test_dispatcher_search_for_candidate(type_a, type_b, type_c, dist, expected_min_dist):
    """
    Checks that _search_candidate finds the item in REF_DICT with the lowest distance to children
    nodes.
    """
    dispatcher = Dispatcher('test')
    dispatcher.REF_DICT = Data.REF_DICT.copy()
    hierarchies = [Data.hierarchy_A, Data.hierarchy_B, Data.hierarchy_C]
    dispatcher.REF_DICT[(type_a, type_b, type_c)] = dist
    min_distance_to_children, _ = dispatcher._search_for_candidate(hierarchies, 0, [])

    assert min_distance_to_children == expected_min_dist
    assert min_distance_to_children == min(dispatcher.REF_DICT.values())


@pytest.mark.parametrize('type_a, type_b, type_c', [
    [RBF, Gaussian, Constant],
    [Matern12, Gaussian, Identity],
    [Matern32, Gaussian, Identity],
])
def test_dispatcher_registered_fn_implemented(type_a, type_b, type_c):
    """
    Checks that the method registered_fn returns the correct (registered) function for given
    argument types.
    """
    dispatcher = Dispatcher('test')

    @dispatcher.register((RBF, Matern12), Gaussian, (Constant, Identity))
    def function_1(kernel, likelihood, mean_function):
        pass

    if type_a is RBF or type_a is Matern12:
        fn = dispatcher.registered_fn(type_a, type_b, type_c)
        assert fn is not None
    else:
        with pytest.raises(NotImplementedError):
            fn = dispatcher.registered_fn(type_a, type_b, type_c)


@pytest.mark.parametrize('type_a, type_b, type_c, expected_fn_name', [
    [RBF, Gaussian, Constant, 'function_1'],
    [RBF, Gaussian, Identity, 'function_1'],
    [Matern12, Gaussian, Identity, 'function_1'],
    [Matern32, Gaussian, Identity, 'base_function'],
])
def test_dispatcher_registered_fn(type_a, type_b, type_c, expected_fn_name):
    """
    Checks that the method registered_fn returns the correct (registered) function for given
    argument types.
    """
    dispatcher = Dispatcher('test')

    @dispatcher.register((RBF, Matern12), Gaussian, (Constant, Identity))
    def function_1(kernel, likelihood, mean_function):
        pass

    @dispatcher.register(Kernel, Likelihood, MeanFunction)
    def base_function(kernel, likelihood, mean_function):
        pass

    fn = dispatcher.registered_fn(type_a, type_b, type_c)
    assert fn.__name__ == expected_fn_name


def test_register_adding_single_type_functions():
    dispatcher = Dispatcher('test')

    @Register(dispatcher, RBF, Gaussian, Constant)
    def function_1(kernel, likelihood, mean_function):
        pass

    @dispatcher.register(RBF, Gaussian, Identity)
    def function_2(kernel, likelihood, mean_function):
        pass

    def function_3(kernel, likelihood, mean_function):
        pass

    Register(dispatcher, Matern12, Likelihood, Identity)(function_3)

    assert len(dispatcher.REF_DICT) == 3
    assert dispatcher.REF_DICT.get((RBF, Gaussian, Constant)) is function_1
    assert dispatcher.REF_DICT.get((RBF, Gaussian, Identity)) is function_2
    assert dispatcher.REF_DICT.get((Matern12, Likelihood, Identity)) is function_3

    with pytest.raises(ValueError):
        @dispatcher.register(RBF, Gaussian, Identity)
        def repeat_function_2(kernel, likelihood, mean_function):
            pass

    with pytest.raises(TypeError):
        def called_function():
            pass

        Register(dispatcher, RBF, Gaussian, Identity)(called_function())


def test_register_breaksdown_multiple_type_functions():
    dispatcher = Dispatcher('test')

    @dispatcher.register((RBF, Matern12), Gaussian, (Constant, Identity))
    def function_1(kernel, likelihood, mean_function):
        pass

    @dispatcher.register(RBF, Gaussian, Linear)
    def function_2(kernel, likelihood, mean_function):
        pass

    assert len(dispatcher.REF_DICT) == 5
    assert dispatcher.REF_DICT.get((RBF, Gaussian, Constant)) is function_1
    assert dispatcher.REF_DICT.get((RBF, Gaussian, Identity)) is function_1
    assert dispatcher.REF_DICT.get((Matern12, Gaussian, Constant)) is function_1
    assert dispatcher.REF_DICT.get((Matern12, Gaussian, Identity)) is function_1
    assert dispatcher.REF_DICT.get((RBF, Gaussian, Linear)) is function_2

    with pytest.raises(ValueError):
        @dispatcher.register(RBF, Gaussian, Identity)
        def repeat_function_2(kernel, likelihood, mean_function):
            pass
