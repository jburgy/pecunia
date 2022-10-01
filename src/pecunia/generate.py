import ast
from collections import defaultdict
from collections.abc import Callable, Generator
from typing import Literal, Optional, Union

import numpy as np

from .atoms import At, Timed
from .code_builder import CodeBuilder

Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
EvolutionStep = Generator[Vector, Optional[tuple[float, Vector, Vector]], None]
Evolution = Callable[[float, Vector, Vector], EvolutionStep]


def from_graph(
    val: Union[At, Timed], implementation: Literal["ast", "bytecode"] = "ast"
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

    return {"ast": _ast, "bytecode": _bytecode}[implementation](m)


def _bytecode(m: defaultdict[float, list]) -> Evolution:
    consts = {None: 0, np.maximum: 1}
    names = {"t": 0, "x": 1, "v": 2}
    b = CodeBuilder(names, consts)
    for time, nodes in sorted(m.items(), reverse=True):
        start_while = len(b)
        b.load_fast("t")
        b.load_const(time)
        b.compare_op(">")
        # anticipate EXTENDED_ARG for start_while
        offset = start_while.bit_length() // 8 + 8
        b.pop_jump_if_false(len(b) + offset * 2)
        b.load_fast("v")
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
            b.yield_value()
            b.unpack_sequence(3)
            b.store_fast("t")
            b.store_fast("x")
            b.store_fast("v")

    b.load_fast("v")
    b.yield_value()

    def evolve(t, x, v):
        t, x, v = yield v

    return b.replace(evolve)


def _ast(m: defaultdict[float, list]):
    communicate = ast.Assign(
        targets=[
            ast.Tuple(
                elts=[
                    ast.Name(id="t", ctx=ast.Store()),
                    ast.Name(id="x", ctx=ast.Store()),
                    ast.Name(id="v", ctx=ast.Store()),
                ],
                ctx=ast.Store(),
            ),
        ],
        value=ast.Yield(value=ast.Name(id="v", ctx=ast.Load())),
    )

    body: list[ast.AST] = []
    for time, nodes in sorted(m.items(), reverse=True):
        body.append(
            ast.While(
                test=ast.Compare(
                    left=ast.Name(id="t", ctx=ast.Load()),
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(value=time)],
                ),
                body=[communicate],
                orelse=[],
            )
        )
        if nodes:
            body.append(
                ast.Assign(
                    targets=[ast.Name(id="v", ctx=ast.Store())],
                    value=nodes[0].expr,
                    type_ignores=[],
                )
            )
            body.append(communicate)

    body.append(ast.Expr(value=ast.Yield(value=ast.Name(id="v", ctx=ast.Load()))))

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
