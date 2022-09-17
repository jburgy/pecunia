from functools import cached_property

import numpy as np

from .code_builder import CodeBuilder


class At:
    def __init__(self, time):
        self.time = time

    @staticmethod
    def __call__(b: CodeBuilder) -> None:
        b.load_fast("x")


class Timed(tuple):
    @cached_property
    def time(self):
        return min(n.time for n in self if hasattr(n, "time"))

    def __repr__(self):
        return type(self).__name__ + super().__repr__()

    def __add__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)


class And(Timed):
    @staticmethod
    def __call__(b: CodeBuilder):
        b.binary_add()


class Or(Timed):
    @staticmethod
    def __call__(b: CodeBuilder):
        b.build_tuple(2)
        b.load_const(np.maximum)
        b.swap()
        b.call_function_ex()
