# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import pytest

from gpflow.kernels import Kernel, Stationary, RBF, Matern12
from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.mean_functions import MeanFunction, Linear, Identity, Constant
from gpflow.util import Dispatcher, Register

class Data:
    REF_DICT = {}
    REF_DICT[('A0', 'B1', 'C1')] = 2
    REF_DICT[('A1', 'B2', 'C0')] = 3
    REF_DICT[('A1', 'B0', 'C0')] = 1

    hierarchy_A = ['A0', 'A1']
    hierarchy_B = ['B0', 'B1', 'B2']
    hierarchy_C = ['C0', 'C1', 'C2']



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


def test_dispatcher_registered_fn(type_a, type_b, type_c, dist, expected_min_dist):
    """
    Checks that the method registered_fn returns the correct (registered) function for given
    argument types.
    """
    dispatcher = Dispatcher('test')
    @dispatcher.register((RBF, Matern12), Gaussian, (Constant,Identity))
    def function_1(kernel, likelihood, mean_function):
        pass

    @dispatcher.register(Kernel, Likelihood, MeanFunction)
    def base_function(kernel, likelihood, mean_function):
        pass

    with pytest.raises(NotImplementedError):
        @dispatcher.register(RBF, Gaussian, Identity)
        def repeat_function_2(kernel, likelihood, mean_function):
            pass



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

    @dispatcher.register((RBF, Matern12), Gaussian, (Constant,Identity))
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




    # def foo(a)
# if __name__ == '__main__':
#     conditional_dispatcher = Dispatcher('conditional')
#
#     @conditional_dispatcher.register(int, str, int)
#     def foo(a, b, c):
#         print(a, b, c)
#
#
#     @conditional_dispatcher.register(str, int, str)
#     def foo2(a, b, c):
#         pass
#
#
#     def foot(a, b, c):
#         foo_fn = conditional_dispatcher.registered_fn(type(a), type(b), type(c))
#         return foo_fn(a, b, c)
#
#
#     print(conditional_dispatcher.REF_DICT)
#     # foot('1',1,'a')
#     foot(0, '1', '1')
#
#     if __name__ == '__main__':
#         conditional_dispatcher = Dispatcher('conditional')
#
#
#         @conditional_dispatcher.register(int, str, (int, float))
#         def foo(a, b, c):
#             print(a, b, c)
#
#
#         @conditional_dispatcher.register(str, int, str)
#         def foo2(a, b, c):
#             pass
#
#
#         def foot(a, b, c):
#             foo_fn = conditional_dispatcher.registered_fn(type(a), type(b), type(c))
#             return foo_fn(a, b, c)
#
#
#         print(conditional_dispatcher.REF_DICT)
#         foot('1', 1, 'a')
#         foot(0, '1', 1)
#
#         #
#         # REF_DICT = {}
#         # REF_DICT[('A0', 'B1', 'C1')] = 'soln_first'
#         # REF_DICT[('A1', 'B2', 'C1')] = 'wrong soln'
#         # REF_DICT[('A1', 'B0', 'C0')] = 'soln_second'
#         #
#         # hierarchy_A = ['A0', 'A1']
#         # hierarchy_B = ['B0', 'B1', 'B2']
#         # hierarchy_C = ['C0', 'C1', 'C2']
#         #
#         #
#         # def search_for_candidate(hierarchies, parent_dist, parent_key, fn=None, dist=None):
#         #     for mro_to_0, parent_0 in enumerate(hierarchies[0]):
#         #         candidate_dist = parent_dist + mro_to_0
#         #         candidate_key = parent_key + [parent_0]
#         #         if len(hierarchies) > 1:
#         #             fn, dist = search_for_candidate(hierarchies[1:], candidate_dist,
#         #                                             candidate_key, fn, dist)
#         #         else:
#         #             # produce candidate
#         #             candidate_fn = REF_DICT.get(tuple(candidate_key), None)
#         #             if candidate_fn is not None:
#         #                 print(candidate_dist, candidate_fn, dist)
#         #                 print(not fn)
#         #                 print((candidate_fn and candidate_dist < dist))
#         #             if not fn or (candidate_fn and candidate_dist < dist):
#         #                 dist = candidate_dist
#         #                 fn = candidate_fn
#         #     return fn, dist
#         #
#         # fn = search_for_candidate([hierarchy_A, hierarchy_B, hierarchy_C], 0, [])
#         # print(fn)
#         a = [[type('a')], 'b', ('c', 'd'), ('e', 'f')]
#         print([key for key in product(*a)])
#
#
#         # key_list = [[]]
#         # tuples_removed = tuple(map(lambda x: x if not isinstance(x, tuple) else None, a))
#         # idx_tuples = [i for i, x in enumerate(a) if isinstance(x, tuple)]
#         # tuples = [list(x) for x in a if isinstance(x, tuple)]
#         # from itertools import product
#         # print(tuples)
#         # for comb in product(list(x) for x in a if isinstance(x, tuple)):
#         #     print(comb)
#         #     # tuples_in = list(tuples_removed)
#         #     # for idx, _ in idx_tuples:
#         #     #     tuples_in[idx] = comb[idx]
#         #     #     key_list.append(tuples_in)
#         #
#         # # for idx, x in enumerate(a):
#         # #     if isinstance(x, tuple):
#         # #         key_list = [key + [j] for key in key_list for j in x]
#         # #     else:
#         # #         key_list = [key + [x] for key in key_list]
#         # print(key_list)
#         def _breakdown_types(*types):
#             return [key for key in product(*types)]
#
#
#         print(_breakdown_types('a', 'b', 'c', 'd', 'e', 'f'))

