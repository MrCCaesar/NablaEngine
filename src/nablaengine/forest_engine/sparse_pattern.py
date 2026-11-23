"""Symbolic sparsity tools for the forest engine beyond nv = 64 (the symbolic analysis; Python only, not
imported by any build path yet).

From a MuJoCo model and its contact pairs this module builds

- the dof tree and the tree pattern of the mass matrix M (entry (i, j) iff one dof is an ancestor of the other),
- the support of every contact pair (dofs on exactly one of the two bodies' root chains; the common ancestors' columns
  cancel at a contact point, and they are cliqued with both branches by the tree pattern anyway) and its class:

    world          one side is the world or welded to it
    chain          one body is an ancestor of the other (the support is a chain: a clique of tree(M))
    robot_object   one side in the robot tree (the tree with the most dofs), the other in another tree ("object")
    object_object  both sides in two different object trees
    self_cheap     a robot self pair across branches whose branch group adds little fill (put in the pattern)
    self_cross     a robot self pair across branches whose group would add much fill (dynamic: handled per world)
    same_object    both sides in one object tree across its branches (treated as self pairs of that tree)

- the elimination order (robot dofs leaves first = reversed dof order, then the object trees, also leaves first),
- the symbolic Cholesky factor of tree(M) + the supports of the pattern pairs + the always-on constraint cliques
  (joint / connect equalities, fixed tendons): nnz(L) (diagonal included), flops, elimination-tree height, the column
  slot tables (column-major in elimination order), and
- per pair the "out of pattern" bit: its support is not a clique of the filled pattern, so a world with that pair
  active cannot use the static sparse factor (fallback / dynamic handling, design section 3.4).

Dof sets are Python ints used as bitmasks (bit i = dof i), so nv is not limited.

    python -m nablaengine.forest_engine.sparse_pattern MODEL.xml     # report (JSON)
    python -m nablaengine.forest_engine.sparse_pattern --selftest    # symbolic / numeric self-tests
"""
import itertools
import json
from dataclasses import dataclass, field

import numpy as np

CLASSES = ("world", "chain", "robot_object", "object_object", "same_object", "self_cheap", "self_cross")
PATTERN_CLASSES = {"world", "chain", "robot_object", "object_object", "self_cheap"}
CHEAP_BUDGET = 64


def popc(x):
    return bin(x).count("1")


def bits(x):
    out, i = [], 0
    while x:
        if x & 1:
            out.append(i)
        x >>= 1
        i += 1
    return out


def mask(it):
    v = 0
    for i in it:
        v |= 1 << int(i)
    return v


@dataclass
class Factor:
    """Symbolic Cholesky factor of a symmetric pattern under an elimination order. Positions are elimination
    positions (perm[k] = original dof eliminated k-th, iperm[dof] = its position); colmask[k] = the positions > k of
    column k's below-diagonal entries; col_start[k] = the slot of column k's diagonal in the column-major storage
    (diagonal first, then the rows in increasing position)."""
    n: int
    perm: list
    iperm: list
    colmask: list
    col_start: list
    etree: list
    nnz: int
    nnz_a: int
    flops: int
    height: int
    filled: list = field(repr=False, default=None)

    def has(self, i, j):
        """(i, j) (original dofs) is an entry of the filled pattern."""
        return i == j or bool(self.filled[i] >> j & 1)

    def slot(self, i, j):
        """Column-major slot of entry (i, j) of L (original dofs); KeyError if it is not in the pattern."""
        p, q = self.iperm[i], self.iperm[j]
        col, row = min(p, q), max(p, q)
        if row == col:
            return self.col_start[col]
        if not self.colmask[col] >> row & 1:
            raise KeyError((i, j))
        return self.col_start[col] + 1 + popc(self.colmask[col] & ((1 << row) - 1))

    def is_clique(self, s):
        """The dof set s (bitmask) is a clique of the filled pattern."""
        for i in bits(s):
            if (s & ~self.filled[i] & ~(1 << i)) != 0:
                return False
        return True

    def summary(self):
        dense = self.n * (self.n + 1) // 2
        return dict(nv=self.n, nnz_a=self.nnz_a, nnz_l=self.nnz, ntri=dense, fill_frac=round(self.nnz / dense, 4),
                    flops=self.flops, dense_flops=dense_flops(self.n), height=self.height)


def dense_flops(n):
    return sum(k * (k + 3) // 2 + 1 for k in range(n))


def symbolic(n, adj, order):
    """Elimination game on the symmetric pattern adj (list of n bitmasks, no diagonal bits needed) in `order`
    (original indices, eliminated first to last). Returns a Factor."""
    assert sorted(order) == list(range(n)), "order must be a permutation"
    adj = [a & ~(1 << i) for i, a in enumerate(adj)]
    for i in range(n):
        for j in bits(adj[i]):
            adj[j] |= 1 << i
    nnz_a = sum(popc(a) for a in adj) // 2 + n
    iperm = [0] * n
    for k, v in enumerate(order):
        iperm[v] = k
    work = list(adj)
    filled = [0] * n
    elim = 0
    colmask, nnz, flops, etree = [], 0, 0, []
    for k, v in enumerate(order):
        nb = work[v] & ~elim & ~(1 << v)
        for u in bits(nb):
            work[u] |= nb & ~(1 << u)
        c = popc(nb)
        nnz += c + 1
        flops += c * (c + 3) // 2 + 1
        cm = mask(iperm[u] for u in bits(nb))
        colmask.append(cm)
        etree.append((cm & -cm).bit_length() - 1 if cm else -1)
        for u in bits(nb):
            filled[v] |= 1 << u
            filled[u] |= 1 << v
        elim |= 1 << v
    col_start, s = [], 0
    for cm in colmask:
        col_start.append(s)
        s += 1 + popc(cm)
    h = [1] * n
    for k in range(n):
        if etree[k] >= 0:
            h[etree[k]] = max(h[etree[k]], h[k] + 1)
    return Factor(n, list(order), iperm, colmask, col_start, etree, nnz, nnz_a, flops, max(h) if n else 0, filled)


def dense_symbolic(n, adj, order):
    """Reference: boolean dense right-looking elimination of the permuted pattern (O(n^3)); returns the lower
    pattern of L in elimination positions as an (n, n) bool array."""
    P = np.zeros((n, n), bool)
    for i in range(n):
        for j in bits(adj[i]):
            P[i, j] = P[j, i] = True
    np.fill_diagonal(P, True)
    o = np.asarray(order)
    A = P[np.ix_(o, o)].copy()
    for k in range(n):
        r = np.nonzero(A[k + 1:, k])[0] + k + 1
        A[np.ix_(r, r)] = True
    return np.tril(A)


def numeric_on_pattern(F, A):
    """Left-looking Cholesky of the dense symmetric A (original indices) computed only on F's slots (float64).
    Returns the values array (column-major slots) and the dense lower L in elimination positions."""
    n = F.n
    o = np.asarray(F.perm)
    Ap = A[np.ix_(o, o)]
    val = np.zeros(F.nnz)
    rows = [[k] + bits(F.colmask[k]) for k in range(n)]
    L = np.zeros((n, n))
    for k in range(n):
        col = {r: Ap[r, k] for r in rows[k]}
        for j in range(k):
            if F.colmask[j] >> k & 1:
                ljk = L[k, j]
                for r in rows[k]:
                    if r == k or F.colmask[j] >> r & 1:
                        col[r] -= L[r, j] * ljk
        d = np.sqrt(col[k])
        for t, r in enumerate(rows[k]):
            L[r, k] = d if r == k else col[r] / d
            val[F.col_start[k] + t] = L[r, k]
    return val, L


class DofTree:
    """Dof / body structure of a MuJoCo model: body root chains as dof masks, trees, robot vs object trees."""

    def __init__(self, m):
        self.m = m
        self.nv, self.nbody = int(m.nv), int(m.nbody)
        self.dof_parent = [int(x) for x in m.dof_parentid]
        self.body_parent = [int(x) for x in m.body_parentid]
        self.body_dofs = [mask(range(int(m.body_dofadr[b]), int(m.body_dofadr[b]) + int(m.body_dofnum[b])))
                          if m.body_dofnum[b] else 0 for b in range(self.nbody)]
        self.chain = [0] * self.nbody
        for b in range(1, self.nbody):
            self.chain[b] = self.chain[self.body_parent[b]] | self.body_dofs[b]
        self.root = [0] * self.nbody
        for b in range(1, self.nbody):
            p = self.body_parent[b]
            self.root[b] = b if p == 0 else self.root[p]
        trees = {}
        for b in range(1, self.nbody):
            trees.setdefault(self.root[b], 0)
            trees[self.root[b]] |= self.body_dofs[b]
        self.trees = {r: d for r, d in trees.items() if d}
        self.robot_root = max(self.trees, key=lambda r: (popc(self.trees[r]), -r)) if self.trees else 0
        self.robot = self.trees.get(self.robot_root, 0)
        self.anc = [0] * self.nv
        for i in range(self.nv):
            p = self.dof_parent[i]
            self.anc[i] = (self.anc[p] | (1 << p)) if p >= 0 else 0

    def static(self, b):
        """Body b does not move (the world or bodies welded to it)."""
        return self.chain[b] == 0

    def tree_adj(self):
        adj = [0] * self.nv
        for i in range(self.nv):
            adj[i] |= self.anc[i]
            for a in bits(self.anc[i]):
                adj[a] |= 1 << i
        return adj

    def order(self):
        """Leaves first, objects last: the robot tree's dofs in reversed dof order, then every other tree's dofs in
        reversed dof order (trees in increasing root order)."""
        rob = [i for i in range(self.nv) if self.robot >> i & 1][::-1]
        objs = []
        for r in sorted(self.trees):
            if r != self.robot_root:
                objs += [i for i in range(self.nv) if self.trees[r] >> i & 1][::-1]
        rest = [i for i in range(self.nv) if i not in set(rob) | set(objs)]
        return rob + objs + rest[::-1]

    def lca(self, b1, b2):
        a = set()
        x = b1
        while True:
            a.add(x)
            if x == 0:
                break
            x = self.body_parent[x]
        x = b2
        while x not in a:
            x = self.body_parent[x]
        return x

    def branch(self, b, lca):
        """The child of lca on the path to b (b itself if its parent is lca)."""
        while self.body_parent[b] != lca:
            b = self.body_parent[b]
        return b


@dataclass
class PairInfo:
    index: int
    geoms: tuple
    bodies: tuple
    cls: str
    support: int
    group: tuple = None
    in_pattern: bool = None

    @property
    def nsup(self):
        return popc(self.support)


def model_pairs(m, include_filter=True):
    """(geom1, geom2) of the explicit <pair>s, then (include_filter) the contype / conaffinity pairs MuJoCo's
    broadphase would add (nablaengine.model.capability.filter_pairs)."""
    out = [(int(a), int(b)) for a, b in zip(m.pair_geom1, m.pair_geom2)]
    if include_filter:
        from nablaengine.model.capability import filter_pairs
        out += [(int(a), int(b)) for a, b in filter_pairs(m)]
    return out


def constraint_cliques(m, t):
    """Dof sets that are always coupled by constraint rows: joint equalities (the two joints' dofs), connect
    equalities (chain xor chain of the two bodies), fixed tendons (their joints' dofs). Returns [(name, mask)]."""
    import mujoco
    out = []
    for e in range(int(m.neq)):
        ty = int(m.eq_type[e])
        o1, o2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if ty == int(mujoco.mjtEq.mjEQ_JOINT):
            s = mask(range(int(m.jnt_dofadr[o1]), int(m.jnt_dofadr[o1]) + 1))
            if o2 >= 0:
                s |= 1 << int(m.jnt_dofadr[o2])
        elif ty in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
            if int(m.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_SITE):
                o1, o2 = int(m.site_bodyid[o1]), int(m.site_bodyid[o2])
            s = t.chain[o1] ^ t.chain[max(o2, 0)]
        else:
            continue
        out.append((f"eq{e}", s))
    for k in range(int(m.ntendon)):
        a, n = int(m.tendon_adr[k]), int(m.tendon_num[k])
        s = 0
        for w in range(a, a + n):
            if int(m.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_JOINT):
                s |= 1 << int(m.jnt_dofadr[int(m.wrap_objid[w])])
        out.append((f"tendon{k}", s))
    return out


class Pattern:
    """The static Hessian pattern P_H of model m with the contact pairs `pairs` ((g1, g2) list; default: the explicit
    pairs and the filter pairs) under the leaves-first / objects-last order."""

    def __init__(self, m, pairs=None, cheap_budget=CHEAP_BUDGET, order=None):
        self.m = m
        t = self.tree = DofTree(m)
        self.pair_list = model_pairs(m) if pairs is None else [tuple(map(int, p)) for p in pairs]
        self.order = list(order) if order is not None else t.order()
        self.cliques = constraint_cliques(m, t)
        self.base_adj = t.tree_adj()
        for _, s in self.cliques:
            self._add(self.base_adj, s)
        self.factor_tree = symbolic(t.nv, t.tree_adj(), self.order)
        self.factor_m = symbolic(t.nv, self.base_adj, self.order)
        self.pairs = [self._classify(k, g1, g2) for k, (g1, g2) in enumerate(self.pair_list)]
        adj = list(self.base_adj)
        for p in self.pairs:
            if p.cls in PATTERN_CLASSES:
                self._add(adj, p.support)
        self.factor_noself = symbolic(t.nv, adj, self.order)
        self.groups = {}
        for p in self.pairs:
            if p.group is not None:
                g = self.groups.setdefault(p.group, dict(pairs=[], support=0))
                g["pairs"].append(p.index)
                g["support"] |= p.support
        base = self.factor_noself.nnz
        cand = []
        for key, g in self.groups.items():
            a2 = list(adj)
            for i in g["pairs"]:
                self._add(a2, self.pairs[i].support)
            g["delta_alone"] = symbolic(t.nv, a2, self.order).nnz - base
            cand.append((g["delta_alone"], key))
        for _, key in sorted(cand):
            g = self.groups[key]
            a2 = list(adj)
            for i in g["pairs"]:
                self._add(a2, self.pairs[i].support)
            nnz = symbolic(t.nv, a2, self.order).nnz
            g["delta"] = nnz - base
            g["cheap"] = g["delta"] <= cheap_budget
            if g["cheap"]:
                adj, base = a2, nnz
            for i in g["pairs"]:
                self.pairs[i].cls = "self_cheap" if g["cheap"] else "self_cross"
        self.adj = adj
        self.factor = symbolic(t.nv, adj, self.order)
        for p in self.pairs:
            p.in_pattern = self.factor.is_clique(p.support)

    @staticmethod
    def _add(adj, s):
        b = bits(s)
        for i in b:
            adj[i] |= s & ~(1 << i)

    def _classify(self, k, g1, g2):
        m, t = self.m, self.tree
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        sup = t.chain[b1] ^ t.chain[b2]
        if t.static(b1) or t.static(b2):
            return PairInfo(k, (g1, g2), (b1, b2), "world", sup)
        r1, r2 = t.root[b1], t.root[b2]
        if r1 != r2:
            rob = (r1 == t.robot_root) or (r2 == t.robot_root)
            return PairInfo(k, (g1, g2), (b1, b2), "robot_object" if rob else "object_object", sup)
        c1, c2 = t.chain[b1], t.chain[b2]
        if (c1 & c2) == c1 or (c1 & c2) == c2:
            return PairInfo(k, (g1, g2), (b1, b2), "chain", sup)
        lca = t.lca(b1, b2)
        br = tuple(sorted((t.branch(b1, lca), t.branch(b2, lca))))
        cls = "self_cross" if r1 == t.robot_root else "same_object"
        return PairInfo(k, (g1, g2), (b1, b2), cls, sup, group=(lca,) + br)

    def out_of_pattern(self):
        """Pair index -> True when the pair's support is not a clique of the static pattern."""
        return {p.index: not p.in_pattern for p in self.pairs}

    def class_counts(self):
        out = {c: 0 for c in CLASSES}
        for p in self.pairs:
            out[p.cls] += 1
        return out

    def report(self, names=True):
        import mujoco
        m = self.m
        bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)
        groups = []
        for key, g in sorted(self.groups.items(), key=lambda kv: kv[1]["delta_alone"]):
            groups.append(dict(lca=bn(key[0]), branches=[bn(key[1]), bn(key[2])], npairs=len(g["pairs"]),
                               nsup=popc(g["support"]), delta_alone=g["delta_alone"], delta=g["delta"],
                               cheap=g["cheap"]))
        return dict(nv=self.tree.nv, nbody=self.tree.nbody, npair=len(self.pairs),
                    trees=len(self.tree.trees), robot_dofs=popc(self.tree.robot),
                    constraint_cliques=len(self.cliques),
                    max_support=max((p.nsup for p in self.pairs), default=0),
                    classes=self.class_counts(),
                    out_of_pattern=sum(not p.in_pattern for p in self.pairs),
                    factor_tree=self.factor_tree.summary(), factor_m=self.factor_m.summary(),
                    factor_no_self=self.factor_noself.summary(), factor_h=self.factor.summary(),
                    factor_all_self=self.with_all().summary(),
                    self_groups=groups if names else len(groups))

    def with_all(self):
        """The factor if every pair (including self_cross) were put into the pattern."""
        adj = list(self.base_adj)
        for p in self.pairs:
            self._add(adj, p.support)
        return symbolic(self.tree.nv, adj, self.order)


def _random_tree(rng, n, nroots):
    par = [-1] * n
    roots = sorted(rng.choice(np.arange(1, n), size=nroots - 1, replace=False).tolist()) if nroots > 1 else []
    for i in range(1, n):
        if i in roots:
            continue
        lo = max(0, i - int(rng.integers(1, 8)))
        par[i] = int(rng.integers(lo, i))
        while par[i] in roots and rng.random() < 0.3:
            par[i] = int(rng.integers(0, i))
    return par


def _anc(par):
    anc = [0] * len(par)
    for i, p in enumerate(par):
        anc[i] = (anc[p] | (1 << p)) if p >= 0 else 0
    return anc


def selftest(seed=0, ntests=200, verbose=False):
    """Symbolic factor vs the dense boolean elimination, tree(M) leaves first has no fill, numeric factor on the
    slots vs numpy's dense Cholesky (float64), out-of-pattern bit vs a direct check. Raises AssertionError."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for it in range(ntests):
        n = int(rng.integers(2, 129 if it % 10 == 0 else 48))
        par = _random_tree(rng, n, int(rng.integers(1, 4)))
        anc = _anc(par)
        adj = [0] * n
        for i in range(n):
            adj[i] |= anc[i]
            for a in bits(anc[i]):
                adj[a] |= 1 << i
        rev = list(range(n))[::-1]
        F0 = symbolic(n, adj, rev)
        assert F0.nnz == F0.nnz_a, ("tree pattern leaves first must not fill", it, F0.nnz, F0.nnz_a)
        cl = []
        for _ in range(int(rng.integers(0, 6))):
            x, y = int(rng.integers(0, n)), int(rng.integers(0, n))
            cl.append((anc[x] | 1 << x) ^ (anc[y] | 1 << y))
        a2 = list(adj)
        for s in cl:
            Pattern._add(a2, s)
        order = rev if rng.random() < 0.7 else rng.permutation(n).tolist()
        F = symbolic(n, a2, order)
        D = dense_symbolic(n, a2, order)
        S = np.zeros((n, n), bool)
        for k in range(n):
            S[k, k] = True
            for r in bits(F.colmask[k]):
                S[r, k] = True
        assert np.array_equal(S, D), ("symbolic != dense symbolic", it)
        assert F.nnz == int(D.sum())
        sl = sorted(F.slot(F.perm[r], F.perm[k]) for k in range(n) for r in range(k, n) if S[r, k])
        assert sl == list(range(F.nnz))
        A = np.zeros((n, n))
        for i in range(n):
            for j in bits(a2[i]):
                if j < i:
                    A[i, j] = A[j, i] = rng.normal()
        A += np.diag(np.abs(A).sum(1) + 1.0 + rng.random(n))
        val, L = numeric_on_pattern(F, A)
        o = np.asarray(F.perm)
        Lref = np.linalg.cholesky(A[np.ix_(o, o)])
        assert np.all(np.abs(Lref[~S]) < 1e-12), ("numpy L has entries outside the pattern", it)
        err = float(np.max(np.abs(L - Lref)))
        res = float(np.max(np.abs(L @ L.T - A[np.ix_(o, o)])) / np.max(np.abs(A)))
        worst = max(worst, err, res)
        assert err < 1e-12 and res < 1e-13, (it, err, res)
        for s in cl + [mask(rng.choice(n, size=min(n, 3), replace=False))]:
            direct = all(F.has(i, j) for i in bits(s) for j in bits(s))
            assert direct == F.is_clique(s)
        if verbose and it % 50 == 0:
            print(it, n, F.summary())
    return dict(tests=ntests, worst_abs=worst)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", nargs="?", help="MJCF / MJB model")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--budget", type=int, default=CHEAP_BUDGET)
    a = ap.parse_args(argv)
    if a.selftest:
        print(json.dumps(selftest(), indent=1))
    if a.model:
        import mujoco
        m = (mujoco.MjModel.from_binary_path(a.model) if a.model.endswith(".mjb")
             else mujoco.MjModel.from_xml_path(a.model))
        print(json.dumps(Pattern(m, cheap_budget=a.budget).report(), indent=1))


if __name__ == "__main__":
    main()
