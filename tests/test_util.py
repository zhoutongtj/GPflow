# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
from gpflow.util import Dispatcher

if __name__ == '__main__':
    conditional_dispatcher = Dispatcher('conditional')

    @conditional_dispatcher.register(int, str, int)
    def foo(a, b, c):
        print(a, b, c)


    @conditional_dispatcher.register(str, int, str)
    def foo2(a, b, c):
        pass


    def foot(a, b, c):
        foo_fn = conditional_dispatcher.registered_fn(type(a), type(b), type(c))
        return foo_fn(a, b, c)


    print(conditional_dispatcher.REF_DICT)
    # foot('1',1,'a')
    foot(0, '1', '1')

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
