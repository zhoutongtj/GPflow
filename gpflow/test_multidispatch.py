from typing import TypeVar, Union
from .multidispatch import Dispatch

dispatch = Dispatch("test", [("a", 0), ("b", 2)])


class A(int):
    pass


class B:
    pass

@dispatch.register(a=float, b=Union[B, float])  # noqa: F811
@dispatch.exclusive_register(a=[A, B], b=int)
def __func(a, c, b):
    return None

@dispatch  # noqa: F811
def __func(a: int, c: object, b: Union[A, B]):
    return a * b

@dispatch.exclusive  # noqa: F811
def __func(a: float, c: object, b: float):
    return a * b

@dispatch.register(a=[int, float], b=[A, B])  # noqa: F811
def __func(a, c: object, b):
    return a * b



print(func(10, 0, 2.))
print(func(10, 0, 2.))