from itertools import accumulate
from opcode import EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, opmap, stack_effect

BINARY_ADD = opmap["BINARY_ADD"]
BUILD_TUPLE = opmap["BUILD_TUPLE"]
CALL_FUNCTION = opmap["CALL_FUNCTION"]
CALL_FUNCTION_EX = opmap["CALL_FUNCTION_EX"]
COMPARE_OP = opmap["COMPARE_OP"]
JUMP_ABSOLUTE = opmap["JUMP_ABSOLUTE"]
LOAD_CONST = opmap["LOAD_CONST"]
LOAD_FAST = opmap["LOAD_FAST"]
POP_JUMP_IF_FALSE = opmap["POP_JUMP_IF_FALSE"]
RETURN_VALUE = opmap["RETURN_VALUE"]
STORE_FAST = opmap["STORE_FAST"]
SWAP = opmap["SWAP" if "SWAP" in opmap else "ROT_TWO"]
UNPACK_SEQUENCE = opmap["UNPACK_SEQUENCE"]
YIELD_VALUE = opmap["YIELD_VALUE"]


class CodeBuilder(bytearray):
    def __init__(self, names, consts):
        self.names = names
        self.consts = consts

    def binary_add(self):
        self.append(BINARY_ADD)
        self.append(0)

    def build_tuple(self, count):
        self.append(BUILD_TUPLE)
        self.append(count)

    def call_function(self, argc=0):
        self.append(CALL_FUNCTION)
        self.append(argc)

    def call_function_ex(self, argc=0):
        self.append(CALL_FUNCTION_EX)
        self.append(argc)

    def compare_op(self, op):
        self.append(COMPARE_OP)
        self.append(cmp_op.index(op))

    def extend_arg(self, arg):
        *ext, arg = arg.to_bytes(arg.bit_length() // 8 + 1, "big")
        for byte in ext:
            self.append(EXTENDED_ARG)
            self.append(byte)
        return arg

    def jump_absolute(self, jabs):
        jabs = self.extend_arg(jabs)
        self.append(JUMP_ABSOLUTE)
        self.append(jabs)

    def load_fast(self, name):
        names = self.names
        self.append(LOAD_FAST)
        self.append(names.setdefault(name, len(names)))

    def load_const(self, const):
        consts = self.consts
        self.append(LOAD_CONST)
        self.append(consts.setdefault(const, len(consts)))

    def pop_jump_if_false(self, jabs):
        jabs = self.extend_arg(jabs)
        self.append(POP_JUMP_IF_FALSE)
        self.append(jabs)

    def return_value(self):
        self.append(RETURN_VALUE)
        self.append(0)

    def store_fast(self, name):
        names = self.names
        self.append(STORE_FAST)
        self.append(names.setdefault(name, len(names)))

    def swap(self):
        self.append(SWAP)
        self.append(0)

    def unpack_sequence(self, count):
        self.append(UNPACK_SEQUENCE)
        self.append(count)

    def yield_value(self):
        self.append(YIELD_VALUE)
        self.append(0)

    def replace(self, func):
        stacksizes = accumulate(
            stack_effect(op) if op < HAVE_ARGUMENT else stack_effect(op, arg)
            for op, arg in zip(self[::2], self[1::2])
        )
        code = func.__code__.replace(
            co_code=bytes(self),
            co_stacksize=max(stacksizes),
            co_firstlineno=1,
            co_names=tuple(self.names),
            co_consts=tuple(self.consts),
            co_filename="generated",
            co_lnotab=b"\x02\x01" * (len(self) // 2),
        )
        return type(func)(code, {})
