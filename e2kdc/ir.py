"""Intermediate representation: typed expressions and statements.

Types are strings: 'b' (bool/predicate), 'i8','i16','i32','i64' (integers,
signedness lives in the operators), 'f32','f64', 'v' (void).
"""

INT_TYPES = ("i8", "i16", "i32", "i64")
BITS = {"b": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64, "f32": 32, "f64": 64, "i128": 128}


def is_int(t):
    return t in INT_TYPES or t == "i128"


def is_float(t):
    return t in ("f32", "f64")


def mask(bits):
    return (1 << bits) - 1


def sext(v, bits):
    v &= mask(bits)
    return v - (1 << bits) if v >> (bits - 1) else v


class Expr:
    ty = "i64"

    def children(self):
        return ()

    def replace(self, fn):
        """Return copy with children mapped through fn (bottom-up done by caller)."""
        return self

    def walk(self):
        """all nodes, pre-order, left to right (a list: an explicit stack is
        several times faster than nested generators on deep trees)"""
        out = []
        stack = [self]
        pop, push, add = stack.pop, stack.extend, out.append
        while stack:
            x = pop()
            add(x)
            ch = x.children()
            if ch:
                push(ch[::-1])
        return out

    def has_side_effects(self):
        return any(isinstance(e, Call) for e in self.walk())

    def has_load(self):
        return any(isinstance(e, (Load, Call)) for e in self.walk())

    def __eq__(self, other):
        return type(self) is type(other) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def key(self):
        return (id(self),)


class Const(Expr):
    def __init__(self, value, ty="i64"):
        self.ty = ty
        if is_int(ty):
            value &= mask(BITS[ty])
        self.value = value

    def key(self):
        return ("C", self.value, self.ty)

    def signed(self):
        if is_int(self.ty):
            return sext(self.value, BITS[self.ty])
        return self.value

    def __repr__(self):
        return "Const(%r,%s)" % (self.value, self.ty)


class FConst(Expr):
    def __init__(self, value, ty="f64"):
        self.value, self.ty = value, ty

    def key(self):
        return ("F", self.value, self.ty)


class Reg(Expr):
    """Machine register (before variable recovery). name canonical: r5, g16, pred0, sp."""

    def __init__(self, name, ty="i64"):
        self.name, self.ty = name, ty

    def key(self):
        return ("R", self.name, self.ty)

    def __repr__(self):
        return "Reg(%s:%s)" % (self.name, self.ty)


class Var(Expr):
    def __init__(self, var, ty=None):
        self.var = var                      # VarInfo object
        self.ty = ty or var.ty

    def key(self):
        return ("V", id(self.var), self.ty)

    def __repr__(self):
        return "Var(%s)" % self.var.name


class Sym(Expr):
    """Address of a symbol plus offset."""

    def __init__(self, name, off=0, kind="data"):
        self.name, self.off, self.kind = name, off, kind
        self.ty = "i64"

    def key(self):
        return ("S", self.name, self.off)

    def __repr__(self):
        return "Sym(%s%+d)" % (self.name, self.off)


class Str(Expr):
    def __init__(self, text):
        self.text = text
        self.ty = "i64"

    def key(self):
        return ("STR", self.text)


class Bin(Expr):
    """Binary op.  ops: + - * /s /u %s %u & | ^ << >>u >>s rotl rotr
    comparisons: == != <s <=s >s >=s <u <=u >u >=u <f <=f ==f !=f >f >=f
    logical: && ||"""

    def __init__(self, op, a, b, ty):
        self.op, self.a, self.b, self.ty = op, a, b, ty

    def children(self):
        return (self.a, self.b)

    def replace(self, fn):
        return Bin(self.op, fn(self.a), fn(self.b), self.ty)

    def key(self):
        return ("B", self.op, self.a.key(), self.b.key(), self.ty)

    def __repr__(self):
        return "(%r %s %r)" % (self.a, self.op, self.b)


class Un(Expr):
    """ops: - ~ ! fneg"""

    def __init__(self, op, a, ty):
        self.op, self.a, self.ty = op, a, ty

    def children(self):
        return (self.a,)

    def replace(self, fn):
        return Un(self.op, fn(self.a), self.ty)

    def key(self):
        return ("U", self.op, self.a.key(), self.ty)


class Cast(Expr):
    """Value conversion.  signed=True means sign extension / signed int<->float."""

    def __init__(self, a, ty, signed=True):
        self.a, self.ty, self.signed = a, ty, signed

    def children(self):
        return (self.a,)

    def replace(self, fn):
        return Cast(fn(self.a), self.ty, self.signed)

    def key(self):
        return ("X", self.a.key(), self.ty, self.signed)

    def __repr__(self):
        return "Cast(%r->%s%s)" % (self.a, "s" if self.signed else "u", self.ty)


class BitCast(Expr):
    def __init__(self, a, ty):
        self.a, self.ty = a, ty

    def children(self):
        return (self.a,)

    def replace(self, fn):
        return BitCast(fn(self.a), self.ty)

    def key(self):
        return ("BC", self.a.key(), self.ty)


class Load(Expr):
    """spec: issued speculatively (`,sm`): an invalid address gives a
    diagnostic value instead of a trap (it is used only under a later check)."""

    def __init__(self, addr, ty, signed=False, spec=False):
        self.addr, self.ty, self.signed, self.spec = addr, ty, signed, spec

    def children(self):
        return (self.addr,)

    def replace(self, fn):
        return Load(fn(self.addr), self.ty, self.signed, self.spec)

    def key(self):
        return ("L", self.addr.key(), self.ty, self.signed)

    def __repr__(self):
        return "Load%s(%r)" % (self.ty, self.addr)


class Select(Expr):
    def __init__(self, c, a, b, ty):
        self.c, self.a, self.b, self.ty = c, a, b, ty

    def children(self):
        return (self.c, self.a, self.b)

    def replace(self, fn):
        return Select(fn(self.c), fn(self.a), fn(self.b), self.ty)

    def key(self):
        return ("SEL", self.c.key(), self.a.key(), self.b.key(), self.ty)


class Call(Expr):
    def __init__(self, target, args, ty="i64"):
        self.target, self.args, self.ty = target, list(args), ty
        self.ret_used = True

    def children(self):
        return (self.target,) + tuple(self.args)

    def replace(self, fn):
        c = Call(fn(self.target), [fn(a) for a in self.args], self.ty)
        # keep what the call site knows (ret_used, info, nfixed, known_arity, tail)
        for k, v in self.__dict__.items():
            if k not in ("target", "args", "ty"):
                c.__dict__[k] = v
        return c

    def key(self):
        return ("CALL", id(self))


class Intr(Expr):
    """Operation without a C equivalent: printed as __e2k_<name>(args)."""

    def __init__(self, name, args, ty="i64", pure=True):
        self.name, self.args, self.ty, self.pure = name, list(args), ty, pure

    def children(self):
        return tuple(self.args)

    def replace(self, fn):
        return Intr(self.name, [fn(a) for a in self.args], self.ty, self.pure)

    def key(self):
        if not self.pure:
            return ("I", id(self))
        return ("I", self.name, tuple(a.key() for a in self.args), self.ty)


def map_expr(e, fn):
    """Bottom-up rewrite."""
    e2 = e.replace(lambda c: map_expr(c, fn))
    return fn(e2)


# ---------------------------------------------------------------- statements
class Stmt:
    guard = None

    def exprs(self):
        return []

    def map(self, fn):
        pass


class Assign(Stmt):
    def __init__(self, dst, src, guard=None, addr=None):
        self.dst, self.src, self.guard, self.addr = dst, src, guard, addr

    def exprs(self):
        return [self.src] + ([self.guard] if self.guard is not None else [])

    def map(self, fn):
        self.src = fn(self.src)
        if self.guard is not None:
            self.guard = fn(self.guard)

    def __repr__(self):
        g = ("if (%r) " % self.guard) if self.guard is not None else ""
        return "%s%r = %r" % (g, self.dst, self.src)


class Store(Stmt):
    def __init__(self, addr, val, ty, guard=None, at=None):
        self.addr, self.val, self.ty, self.guard, self.at = addr, val, ty, guard, at

    def exprs(self):
        return [self.addr, self.val] + ([self.guard] if self.guard is not None else [])

    def map(self, fn):
        self.addr, self.val = fn(self.addr), fn(self.val)
        if self.guard is not None:
            self.guard = fn(self.guard)

    def __repr__(self):
        return "store%s[%r] = %r" % (self.ty, self.addr, self.val)


class ExprStmt(Stmt):
    """Expression evaluated for side effects (call, intrinsic)."""

    def __init__(self, e, guard=None, addr=None):
        self.e, self.guard, self.addr = e, guard, addr

    def exprs(self):
        return [self.e] + ([self.guard] if self.guard is not None else [])

    def map(self, fn):
        self.e = fn(self.e)
        if self.guard is not None:
            self.guard = fn(self.guard)

    def __repr__(self):
        return "eval %r" % (self.e,)


class Comment(Stmt):
    def __init__(self, text):
        self.text = text


# block terminators
class Jump(Stmt):
    def __init__(self, target):
        self.target = target


class Branch(Stmt):
    def __init__(self, cond, t, f):
        self.cond, self.t, self.f = cond, t, f

    def exprs(self):
        return [self.cond]

    def map(self, fn):
        self.cond = fn(self.cond)


class Return(Stmt):
    def __init__(self, value=None):
        self.value = value

    def exprs(self):
        return [self.value] if self.value is not None else []

    def map(self, fn):
        if self.value is not None:
            self.value = fn(self.value)


class IJump(Stmt):
    def __init__(self, target):
        self.target = target

    def exprs(self):
        return [self.target]

    def map(self, fn):
        self.target = fn(self.target)


class Switch(Stmt):
    def __init__(self, index, targets):
        self.index, self.targets = index, list(targets)

    def exprs(self):
        return [self.index]

    def map(self, fn):
        self.index = fn(self.index)


class Unreachable(Stmt):
    def __init__(self, why=""):
        self.why = why
