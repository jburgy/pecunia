import ast
from functools import cached_property

import numpy as np

from .code_builder import CodeBuilder


class At:
    def __init__(self, time):
        self.time = time

    @staticmethod
    def __call__(b: CodeBuilder) -> None:
        b.load_fast("x")

    def expr(self, names):
        return ast.Name(id="x", ctx=ast.Load())


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

    def expr(self, names):
        value = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()), attr="add", ctx=ast.Load()
            ),
            args=[
                arg.expr(names) if hasattr(arg, "expr") else ast.Constant(value=arg)
                for arg in self
            ],
            keywords=[],
        )
        return (
            value
            if (name := names.get(self)) is None
            else ast.NamedExpr(
                target=ast.Name(id=name, ctx=ast.Store()), value=value, type_ignores=[]
            )
        )


class Or(Timed):
    @staticmethod
    def __call__(b: CodeBuilder):
        b.build_tuple(2)
        b.load_const(np.maximum)
        b.swap()
        b.call_function_ex()

    def expr(self, names):
        value = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()), attr="maximum", ctx=ast.Load()
            ),
            args=[
                arg.expr(names) if hasattr(arg, "expr") else ast.Constant(value=arg)
                for arg in self
            ],
            keywords=[],
        )
        return (
            value
            if (name := names.get(self)) is None
            else ast.NamedExpr(
                target=ast.Name(id=name, ctx=ast.Store()), value=value, type_ignores=[]
            )
        )
