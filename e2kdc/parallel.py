"""Per-function passes in worker processes.

The two per-function passes of Decompiler.decompile (summaries, then
lifting + analysis + structuring) are independent between functions, so they
run in forked workers (CPython threads would share one interpreter lock).
The output does not depend on the number of workers:

* every worker is a fork of the parent taken just before the pass, so it sees
  the same summaries and context;
* results come back pickled and are used in the original function order;
* the one piece of shared state the second pass writes, the prototypes of
  called functions (Context.add_proto, order-dependent), is recorded in the
  worker and replayed by the parent in function order;
* the objdump functions (and the context) are passed by reference
  (persistent ids), not copied, so the parent's objects are the ones the
  results point to;
* a function whose worker died or whose result cannot be pickled is redone in
  the parent, at its place in the order.
"""
import io
import os
import pickle
import sys

_JOB = None          # what the forked workers run: (decompiler, method, extra, protos)


def cpus():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _can_fork():
    return hasattr(os, "fork") and _JOB is None      # no nested pools


def _shared(dec):
    """objects the results may point to that must stay the parent's: id -> pid"""
    ids = {id(f): ("f", i) for i, f in enumerate(dec.funcs)}
    for name in ("ctx", "res", "elf"):
        o = getattr(dec, name, None)
        if o is not None:
            ids[id(o)] = ("a", name)
    return ids


class _Pickler(pickle.Pickler):
    def __init__(self, f, ids):
        super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
        self.ids = ids

    def persistent_id(self, o):
        return self.ids.get(id(o))


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, dec):
        super().__init__(f)
        self.dec = dec

    def persistent_load(self, pid):
        if pid[0] == "f":
            return self.dec.funcs[pid[1]]
        return getattr(self.dec, pid[1])


def _deep():
    if sys.getrecursionlimit() < 20000:     # pickling nests as deep as the trees
        sys.setrecursionlimit(20000)


def _work(fi):
    dec, method, extra, protos = _JOB
    if protos:
        dec.ctx.proto_log = []
    try:
        r = getattr(dec, method)(dec.funcs[fi], *extra)
    finally:
        log = dec.ctx.proto_log or []
        dec.ctx.proto_log = None
    buf = io.BytesIO()
    _Pickler(buf, _shared(dec)).dump((r, log))
    return buf.getvalue()


def _read(fd, n):
    out = bytearray()
    while len(out) < n:
        b = os.read(fd, min(n - len(out), 1 << 20))
        if not b:
            return None
        out += b
    return bytes(out)


def _write(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _child(tr, rw):
    """worker loop: task index in, result out (length 0: failed)"""
    _deep()
    while True:
        h = _read(tr, 4)
        if h is None:
            return
        fi = int.from_bytes(h, "little", signed=True)
        if fi < 0:
            return
        try:
            blob = _work(fi)
        except BaseException:
            blob = b""
        _write(rw, len(blob).to_bytes(8, "little") + blob)


def run(dec, method, items, extra, protos=False):
    """[dec.<method>(f, *extra) for f in items], in worker processes when
    dec.jobs allows (0: one per CPU).  items: functions of dec.funcs.
    protos: the method adds prototypes (Context.add_proto) to replay."""
    global _JOB
    import select
    fn = getattr(dec, method)
    jobs = dec.jobs if dec.jobs else cpus()
    jobs = min(jobs, len(items))
    if jobs <= 1 or not _can_fork():
        return [fn(f, *extra) for f in items]
    index = {id(f): i for i, f in enumerate(dec.funcs)}
    # the biggest functions first: the last ones to finish are then short
    todo = sorted(range(len(items)), key=lambda k: len(items[k].bundles))
    blobs = {}
    kids = {}                       # result fd -> [pid, task fd, current item or None]
    sys.stdout.flush()
    sys.stderr.flush()
    _JOB = (dec, method, extra, protos)
    try:
        for _ in range(jobs):
            tr, tw = os.pipe()
            rr, rw = os.pipe()
            pid = os.fork()
            if pid == 0:                        # worker
                code = 0
                try:
                    os.close(tw)
                    os.close(rr)
                    for fd in list(kids):       # pipes of the workers forked before
                        os.close(fd)
                        os.close(kids[fd][1])
                    _child(tr, rw)
                except BaseException:
                    code = 1
                finally:
                    os._exit(code)
            os.close(tr)
            os.close(rw)
            kids[rr] = [pid, tw, None]

        def give(fd):
            k = kids[fd]
            if todo:
                k[2] = todo.pop()
                msg = index[id(items[k[2]])]
            else:
                k[2] = None
                msg = -1
            try:
                _write(k[1], msg.to_bytes(4, "little", signed=True))
            except OSError:                     # the worker died: its item is redone here
                k[2] = None
        for fd in list(kids):
            give(fd)
        live = {fd for fd in kids if kids[fd][2] is not None}
        while live:
            ready, _, _ = select.select(list(live), [], [])
            for fd in ready:
                h = _read(fd, 8)
                blob = None if h is None else _read(fd, int.from_bytes(h, "little"))
                if blob is None:                # the worker died
                    live.discard(fd)
                    continue
                blobs[kids[fd][2]] = blob or None
                give(fd)
                if kids[fd][2] is None:
                    live.discard(fd)
    finally:
        _JOB = None
        for fd, (pid, tw, _) in kids.items():
            for x in (fd, tw):
                try:
                    os.close(x)
                except OSError:
                    pass
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
    _deep()
    out = []
    for k, f in enumerate(items):
        blob = blobs.get(k)
        if blob is None:
            out.append(fn(f, *extra))           # redone here; applies its prototypes directly
            continue
        r, log = _Unpickler(io.BytesIO(blob), dec).load()
        for ev in log:
            dec.ctx.add_proto(*ev)
        out.append(r)
    return out
