from collections import defaultdict
from collections.abc import Callable, Generator
from typing import Optional, Union

import numpy as np

from .atoms import At, Timed
from .code_builder import CodeBuilder

Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
EvolutionStep = Generator[Vector, Optional[tuple[float, Vector, Vector]], None]
Evolution = Callable[[float, Vector, Vector], EvolutionStep]


def from_graph(val: Union[At, Timed]) -> Evolution:
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
