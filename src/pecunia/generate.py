import ast
from collections import defaultdict
from collections.abc import Callable, Generator, ValuesView
from itertools import chain
from typing import Literal, Optional, Union

import numpy as np

from .atoms import At, Timed
from .code_builder import CodeBuilder

Value = Union[At, Timed]
Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
EvolutionStep = Generator[Vector, Optional[tuple[float, Vector, Vector]], None]
Evolution = Callable[[float, Vector, Vector], EvolutionStep]


def from_graph(
    val: Value, implementation: Literal["ast", "bytecode"] = "ast"
) -> Evolution:
    m: defaultdict[float, list] = defaultdict(list)
    m[0.0]

    q = [val]
    while q:
        n = q.pop()
        try:
            time = n.time
        except AttributeError:
            pass
        m[time].append(n)
        try:
            q.extend(n)  # type: ignore
        except TypeError:
            pass

    return {"ast": _ast, "bytecode": _bytecode}[implementation](val, m)


def _bytecode(val: Value, m: defaultdict[float, list]) -> Evolution:
    consts = {None: 0, np.maximum: 1}
    names = {"t": 0, "x": 1, "v": 2}
    b = CodeBuilder(names, consts)
    for time, nodes in sorted(m.items(), reverse=True):
        start_while = len(b)
        b.load_fast("t")
        b.load_const(time)
        b.compare_op(">")
        # anticipate EXTENDED_ARG for start_while
        offset = start_while.bit_length() // 8 + 9
        b.pop_jump_if_false(len(b) + offset * 2)
        b.load_fast("v")
        b.build_tuple(1)
        b.yield_value()
        b.unpack_sequence(3)
        b.store_fast("t")
        b.store_fast("x")
        b.store_fast("v")
        b.jump_absolute(start_while)
        for node in reversed(nodes):
            if callable(node):
                node(b)
            else:
                b.load_const(node)
        if nodes:
            b.build_tuple(1)
            b.yield_value()
            b.unpack_sequence(3)
            b.store_fast("t")
            b.store_fast("x")
            b.store_fast("v")

    b.load_fast("v")
    b.build_tuple(1)
    b.yield_value()

    def evolve(t, x, v):
        t, x, v = yield v

    return b.replace(evolve)


def _analyze(m: defaultdict[float, list]):
    earliest = {}
    for time, nodes in sorted(m.items(), reverse=True):
        for node in nodes:
            try:
                for arg in node:
                    if hasattr(arg, "time"):
                        earliest[arg] = time
            except TypeError:
                pass
    return earliest


def _communicate(names: ValuesView[str]):
    store = ast.Store()
    load = ast.Load()
    return ast.Assign(
        targets=[
            ast.Tuple(
                elts=[
                    ast.Name(id="t", ctx=store),
                    ast.Name(id="x", ctx=store),
                    *(ast.Name(id=name, ctx=store) for name in names),
                ],
                ctx=store,
            ),
        ],
        value=ast.Yield(
            value=ast.Tuple(
                elts=[ast.Name(id=name, ctx=load) for name in names],
                ctx=load,
            )
        ),
    )


def _ast(val: Value, m: defaultdict[float, list]):
    earliest = _analyze(m)
    earliest[val] = 0.0

    body: list[ast.AST] = []
    names: dict[Union[At, Timed], str] = {}
    for time, nodes in sorted(m.items(), reverse=True):
        body.append(
            ast.While(
                test=ast.Compare(
                    left=ast.Name(id="t", ctx=ast.Load()),
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(value=time)],
                ),
                body=[_communicate(names.values())],
                orelse=[],
            )
        )
        if nodes:
            temp = chain(names, nodes)
            names = {}
            for node in temp:
                until = earliest.get(node)
                if until is not None and until < time:
                    names.setdefault(node, f"v{len(names)}")
            body.append(ast.Expr(value=nodes[0].expr(names)))
            body.append(_communicate(names.values()))

    body.append(
        ast.Expr(
            value=ast.Yield(
                value=ast.Tuple(
                    elts=[ast.Name(id=names[val], ctx=ast.Load())], ctx=ast.Load()
                )
            )
        )
    )

    evolve = ast.fix_missing_locations(
        ast.FunctionDef(
            name="evolve",
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg="t"),
                    ast.arg(arg="x"),
                    ast.arg(arg="v"),
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            lineno=1,
        )
    )
    g = {"np": np}
    exec(ast.unparse(evolve), g)
    return g["evolve"]
