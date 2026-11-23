"""Collision mid-phase of the forest engine: skip whole groups of contact pairs whose bounding spheres are apart."""
import json
import os

import mujoco
import numpy as np

CHAIN = 4
PAD_REL, PAD_ABS = 1e-4, 1e-4
TEXT = "nablaengine_midphase"


def mode():
    v = os.environ.get("DK_MIDPHASE", "auto").strip().lower() or "auto"
    if v not in ("auto", "0", "1", "2"):
        raise ValueError(f"DK_MIDPHASE={v!r}: auto, 0, 1 or 2")
    return v


def group_level(t):
    """Whether a build gets the group level: $DK_MIDPHASE=2, or a model with explicit groups (its text element)."""
    return mode() == "2" or (mode() != "1" and t["explicit"])


def _explicit(m):
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXT, TEXT)
    if k < 0:
        return []
    raw = bytes(m.text_data[m.text_adr[k]:m.text_adr[k] + m.text_size[k] - 1]).decode()
    out = []
    for names in json.loads(raw):
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        bad = [n for n, i in zip(names, ids) if i <= 0]
        if bad:
            raise ValueError(f"{TEXT}: unknown or world bodies {bad}")
        out.append(ids)
    return out


def groups(m, colbodies):
    """Body -> group root (an id among the bodies; world bodies -1) by the default rule, explicit groups first."""
    nb = m.nbody
    par = np.asarray(m.body_parentid)
    has = np.zeros(nb, bool)
    for b in sorted(colbodies, reverse=True):
        has[b] = True
    for b in range(nb - 1, 0, -1):
        if has[b]:
            has[par[b]] = True
    kids = [[] for _ in range(nb)]
    for b in range(1, nb):
        if has[b]:
            kids[par[b]].append(b)

    def chain_len(b):
        n = 0
        while True:
            n += 1
            if len(kids[b]) > 1:
                return None
            if not kids[b]:
                return n
            b = kids[b][0]

    root = np.full(nb, -1)
    for gi, ids in enumerate(_explicit(m)):
        for b in ids:
            if root[b] >= 0:
                raise ValueError(f"{TEXT}: body {m.body(b).name} in two groups")
            root[b] = nb + gi
    for b in range(1, nb):
        if root[b] >= 0:
            continue
        p = par[b]
        hand = p > 0 and len(kids[p]) >= 2 and all((c := chain_len(k)) is not None and c <= CHAIN for k in kids[p])
        phand = p > 0 and root[p] == p and _is_hand(p, kids, chain_len)
        if p == 0 or (len(kids[p]) >= 2 and not hand and not phand) or (root[p] >= nb):
            root[b] = b
        elif _is_hand(b, kids, chain_len):
            root[b] = b
        else:
            root[b] = root[p]
    return root


def _is_hand(b, kids, chain_len):
    return len(kids[b]) >= 2 and all((c := chain_len(k)) is not None and c <= CHAIN for k in kids[b])


def tables(m):
    """The mid-phase tables of model m (dict), or None for a model without pairs or beyond the table limits (255
    nodes, 255 groups)."""
    from nablaengine.forest_engine import step
    if m.npair == 0:
        return None
    cg = step.collision_geoms(m)
    table = {g: i for i, g in enumerate(cg)}
    gb = np.asarray(m.geom_bodyid)
    fix = [-1] * len(cg)
    for i, f in enumerate(step.fixture_groups(m)):
        for g in f["geoms"]:
            fix[g] = i
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    node_of, nodes = {}, []
    for i, g in enumerate(cg):
        key = ("g", g) if gb[g] == 0 or m.geom_type[g] == plane else ("b", int(gb[g]))
        if key not in node_of:
            node_of[key] = len(nodes)
            nodes.append((key, []))
        nodes[node_of[key]][1].append(i)
    if len(nodes) > 255:
        return None
    nword, off = [], []
    for key, gi in nodes:
        assert len(set(fix[i] for i in gi)) == 1, "mid-phase node: fixture and non-fixture geoms on one body"
        gs = [cg[i] for i in gi]
        ref = gs[0]
        isp = m.geom_type[ref] == plane
        if isp:
            assert len(gs) == 1
            nword.append(gi[0] | 1 << 8)
            off.append((0.0, 0.0, 0.0, 0.0))
            continue
        pos = np.asarray([m.geom_pos[g] for g in gs], np.float64)
        rb = np.asarray([m.geom_rbound[g] for g in gs], np.float64)
        c = ((pos - rb[:, None]).min(0) + (pos + rb[:, None]).max(0)) / 2
        r = float(max(np.linalg.norm(pos - c, axis=1) + rb))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(m.geom_quat[ref], np.float64))
        o = R.reshape(3, 3).T @ (c - m.geom_pos[ref])
        nword.append(gi[0])
        off.append((*o, r * (1 + PAD_REL) + PAD_ABS))
    colbodies = sorted({int(gb[g]) for g in cg if gb[g] != 0})
    root = groups(m, colbodies)
    gkey, group_of_node = {}, []
    for (key, gi) in nodes:
        k = ("n", key) if key[0] == "g" else ("r", int(root[key[1]]))
        if k not in gkey:
            gkey[k] = len(gkey)
        group_of_node.append(gkey[k])
    ngrp = len(gkey)
    if ngrp > 255:
        return None
    gnodes = [[n for n in range(len(nodes)) if group_of_node[n] == gi] for gi in range(ngrp)]
    gfirst = np.cumsum([0] + [len(x) for x in gnodes]).tolist()
    runs, rstart = [], []
    for p in range(m.npair):
        g1, g2 = step.pair_order(m, int(m.pair_geom1[p]), int(m.pair_geom2[p]))
        k1 = ("g", g1) if gb[g1] == 0 or m.geom_type[g1] == plane else ("b", int(gb[g1]))
        k2 = ("g", g2) if gb[g2] == 0 or m.geom_type[g2] == plane else ("b", int(gb[g2]))
        r = (node_of[k1], node_of[k2])
        if not runs or runs[-1] != r:
            runs.append(r)
            rstart.append(p)
    rstart.append(m.npair)
    gp_of, gp_runs = {}, []
    for ri, (n1, n2) in enumerate(runs):
        a, b = group_of_node[n1], group_of_node[n2]
        pa, pb = bool(nword[gnodes[a][0]] >> 8), bool(nword[gnodes[b][0]] >> 8)
        assert not pb, "mid-phase: a plane as the second geom of a pair"
        k = (a, b) if pa or a <= b else (b, a)
        if k not in gp_of:
            gp_of[k] = len(gp_of)
            gp_runs.append([])
        gp_runs[gp_of[k]].append(ri)
    gpw = []
    for (a, b) in gp_of:
        pa = nword[gnodes[a][0]] >> 8
        gpw.append(a | b << 8 | (pa << 16) | ((nword[gnodes[a][0]] & 255) << 17 if pa else 0))
    rword = [n1 | n2 << 8 | ((nword[n1] >> 8) << 16 | (nword[n1] & 255) << 17 if nword[n1] >> 8 else 0)
             for n1, n2 in runs]
    gpfirst = np.cumsum([0] + [len(x) for x in gp_runs]).tolist()
    return dict(nodes=nword, off=off, gfirst=gfirst, gnodes=sum(gnodes, []), gp=gpw, runs=rword, gpfirst=gpfirst,
                gpruns=sum(gp_runs, []), rstart=rstart, explicit=bool(_explicit(m)), names=[_name(m, k, cg) for k, _ in nodes],
                group_of_node=group_of_node)


def _name(m, key, cg):
    return m.geom(key[1]).name or f"geom{key[1]}" if key[0] == "g" else m.body(key[1]).name


def enabled(m):
    """Whether model m's forest build gets the mid-phase ($DK_MIDPHASE, module docstring)."""
    v = mode()
    if v == "0" or m.npair == 0:
        return False
    if v == "auto" and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXT, TEXT) < 0:
        return False
    return tables(m) is not None


def describe(m):
    """Human-readable groups of model m: [(group, [node names])] (the default rule plus explicit groups)."""
    t = tables(m)
    if t is None:
        return []
    out = {}
    for n, g in enumerate(t["group_of_node"]):
        out.setdefault(g, []).append(t["names"][n])
    return sorted(out.items())


def _arr(name, kind, vals):
    return f"__device__ const {kind} {name}[{max(len(vals), 1)}] = {{{', '.join(map(str, vals or [0]))}}};\n"


def header(m, t=None):
    """The mid-phase's #defines and tables (replaces step._groups_header's text when enabled)."""
    from nablaengine.forest_engine import step
    t = t or tables(m)
    f32 = lambda x: repr(float(np.float32(x))) + "f"
    nrun = len(t["runs"])
    groups = group_level(t)
    return (f"#define BROAD_GROUPS 0\n#define MIDPHASE 1  // the collision mid-phase (forest_engine.midphase)\n"
            f"#define MP_NNODE {len(t['nodes'])}\n#define MP_NGRP {len(t['gfirst']) - 1}\n#define MP_NGP {len(t['gp'])}\n"
            f"#define MP_NRUN {nrun}\n#define MP_NRW {(nrun + 31) // 32}\n#define NCAND {min(step.NCAND, m.npair)}\n"
            f"#define MP_GROUPS {int(groups)}  // the group level (midphase.group_level)\n"
            f"#define MP_SCRATCH {scratch(m, t, groups)}  // (midphase.scratch)\n"
            + _arr("c_mpnode", "unsigned char", [w & 255 for w in t["nodes"]])
            + _arr("c_mpoff", "float", [f32(x) for o in t["off"] for x in o])
            + _arr("c_mpgfirst", "unsigned short", t["gfirst"])
            + _arr("c_mpgnode", "unsigned char", t["gnodes"])
            + _arr("c_mpgp", "unsigned", t["gp"])
            + _arr("c_mprun", "unsigned", t["runs"])
            + (_arr("c_mpgpfirst", "unsigned short", t["gpfirst"]) + _arr("c_mpgpruns", "unsigned short", t["gpruns"])
               if groups else "")
            + _arr("c_mprstart", "unsigned short", t["rstart"]))


def scratch(m, t, groups):
    """The collision's shared-memory scratch (the MP_SCRATCH member list, in place of step.py's cand[NPAIR]). Where it
    fits, it lives inside the candidate array, so the structs do not grow (a larger SimP costs k_pre a world per SM:
    V100 G1 + rigid hands 11 -> 10, k_pre +15%): the node spheres (and the group level's arrays) at its head, which the
    final pass overwrites with candidates only after the runs were tested, and the kept-run list at its tail, starting
    at NPAIR - NRUN: the candidates written before kept run j is read come from the kept runs before it, at most
    NPAIR - NRUN + j pairs (every run has at least one pair), so they never reach it. Without the room (or with a
    candidate capacity below NPAIR) the arrays are separate."""
    from nablaengine.forest_engine import step
    nn, ng, ngp, nrun = len(t["nodes"]), len(t["gfirst"]) - 1, len(t["gp"]), len(t["runs"])
    ncand = min(step.NCAND, m.npair)
    head = ["float mpn[MP_NNODE][4];"]
    hbytes = 16 * nn
    if groups:
        head += ["float mpg[MP_NGRP][4];", "unsigned mprm[MP_NRW];", "unsigned short mpk[MP_NGP];"]
        hbytes += 16 * ng + 4 * ((nrun + 31) // 32) + 2 * ngp
    hbytes += hbytes % 4
    gap = ncand - nrun - hbytes // 2
    if ncand == m.npair and gap >= 1:
        return ("union { unsigned short cand[NCAND]; struct { " + " ".join(head)
                + f" unsigned short mp_gap[{gap}], mprl[MP_NRUN]; }}; }};")
    return " ".join(head) + " unsigned short mprl[MP_NRUN], cand[NCAND];"

HELPER = r"""// the mid-phase's sphere test (forest_engine.midphase): spheres A, B (centre, radius) overlap, or with pg > 0 A is the
// plane of geom pg - 1 (A: a point on it) and B's sphere reaches below it; conservative: the radii are padded
template <class S>
__device__ __forceinline__ bool mp_overlap(const S& s, const float* A, const float* B, int pg) {
  float d[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]};
  if (pg) {
    float n[3];
    zaxis_rn(s.gq[pg - 1], n);
    return d[0] * n[0] + d[1] * n[1] + d[2] * n[2] <= B[3];
  }
  const float rb = A[3] + B[3];
  return d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= rb * rb;
}

"""

DEVICE = r"""#if MIDPHASE
    // the mid-phase (forest_engine.midphase): the node spheres (fixed in a geom's frame); with the group level the
    // group spheres (around their nodes' bounding box) and the group pairs' overlap bits; the runs whose group pair
    // overlaps and whose node spheres overlap, in run order = pair order; and their pairs through the per-pair test
    // into s.cand: the candidates of the plain loop below, in the same order
    for (int k = lane; k < MP_NNODE; k += 32) {
      const int g = c_mpnode[k];
      const float* o = c_mpoff + 4 * k;
      float t[3];
      qrot_rn(s.gq[g], o, t);
      for (int i = 0; i < 3; i++) s.mpn[k][i] = s.gx[g][i] + t[i];
      s.mpn[k][3] = o[3];
    }
    int nkr = 0;
#if MP_GROUPS
    for (int i = lane; i < MP_NRW; i += 32) s.mprm[i] = 0u;
    __syncwarp();
    for (int k = lane; k < MP_NGRP; k += 32) {
      const int a = c_mpgfirst[k], b = c_mpgfirst[k + 1];
      float c[3], r;
      if (b - a == 1) {
        const int n = c_mpgnode[a];
        for (int i = 0; i < 3; i++) c[i] = s.mpn[n][i];
        r = s.mpn[n][3];
      } else {
        float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (int j = a; j < b; j++) {
          const int n = c_mpgnode[j];
          for (int i = 0; i < 3; i++) {
            lo[i] = fminf(lo[i], s.mpn[n][i] - s.mpn[n][3]);
            hi[i] = fmaxf(hi[i], s.mpn[n][i] + s.mpn[n][3]);
          }
        }
        float h[3];  // the box's centre and half diagonal: a sphere that contains the box, so every node sphere
        for (int i = 0; i < 3; i++) { c[i] = 0.5f * (lo[i] + hi[i]); h[i] = 0.5f * (hi[i] - lo[i]); }
        r = sqrtf(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]) * (1.f + 1e-5f) + 1e-6f;  // (+ the rounding of the above)
      }
      for (int i = 0; i < 3; i++) s.mpg[k][i] = c[i];
      s.mpg[k][3] = r;
    }
    __syncwarp();
    int nk = 0;
    for (int k0 = 0; k0 < MP_NGP; k0 += 32) {  // the group pairs, one per lane: the overlapping ones into s.mpk
      const int k = k0 + lane;
      bool keep = false;
      if (k < MP_NGP) {
        const unsigned e = c_mpgp[k];
        keep = mp_overlap(s, s.mpg[e & 255u], s.mpg[(e >> 8) & 255u], (e >> 16) & 1u ? (int)(e >> 17) + 1 : 0);
      }
      const unsigned bal = __ballot_sync(0xffffffffu, keep);
      if (keep) s.mpk[nk + __popc(bal & lanes_below)] = (unsigned short)k;
      nk += __popc(bal);
    }
    __syncwarp();
    for (int j0 = 0; j0 < nk; j0 += 32) {  // the kept group pairs' runs (c_mpgpruns: the runs grouped by group pair;
      // lane j: group pair j's runs at stream positions exc_j .. inc_j - 1), a kept run's bit set in s.mprm
      const int j = j0 + lane, gp = j < nk ? s.mpk[j] : 0;
      const int first = j < nk ? c_mpgpfirst[gp] : 0, cnt = j < nk ? c_mpgpfirst[gp + 1] - first : 0;
      int inc = cnt;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
      }
      const int total = __shfl_sync(0xffffffffu, inc, 31), exc = inc - cnt;
      for (int t0 = 0; t0 < total; t0 += 32) {
        const int t = t0 + lane;
        int lo = 0;
#pragma unroll
        for (int st = 16; st; st >>= 1) {
          int e = __shfl_sync(0xffffffffu, exc, (lo + st) & 31);
          if (lo + st < 32 && e <= t) lo += st;
        }
        const int at = __shfl_sync(0xffffffffu, first, lo) + t - __shfl_sync(0xffffffffu, exc, lo);
        if (t < total) {
          const int ri = c_mpgpruns[at];
          const unsigned e = c_mprun[ri];
          if (mp_overlap(s, s.mpn[e & 255u], s.mpn[(e >> 8) & 255u], (e >> 16) & 1u ? (int)(e >> 17) + 1 : 0))
            atomicOr(&s.mprm[ri >> 5], 1u << (ri & 31));
        }
      }
    }
    __syncwarp();
    for (int w0 = 0; w0 < MP_NRW; w0 += 32) {  // the kept runs in run order (= pair order)
      const unsigned word = w0 + lane < MP_NRW ? s.mprm[w0 + lane] : 0u;
      const int c = __popc(word);
      int inc = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
      }
      int pos = nkr + inc - c;
      for (unsigned x = word; x; x &= x - 1u) s.mprl[pos++] = (unsigned short)((w0 + lane) * 32 + __ffs(x) - 1);
      nkr += __shfl_sync(0xffffffffu, inc, 31);
    }
#else
    __syncwarp();
#pragma unroll 4
    for (int r0 = 0; r0 < MP_NRUN; r0 += 32) {  // the runs, one per lane, the kept ones in run order (= pair order)
      const int ri = r0 + lane;
      bool keep = false;
      if (ri < MP_NRUN) {
        const unsigned e = c_mprun[ri];
        keep = mp_overlap(s, s.mpn[e & 255u], s.mpn[(e >> 8) & 255u], (e >> 16) & 1u ? (int)(e >> 17) + 1 : 0);
      }
      const unsigned bal = __ballot_sync(0xffffffffu, keep);
      if (keep) s.mprl[nkr + __popc(bal & lanes_below)] = (unsigned short)ri;
      nkr += __popc(bal);
    }
#endif
    __syncwarp();
    for (int j0 = 0; j0 < nkr; j0 += 32) {  // their pairs (lane j: run j's pairs at stream positions exc_j .. inc_j - 1)
      const int j = j0 + lane, ri = j < nkr ? s.mprl[j] : 0;
      const int first = j < nkr ? c_mprstart[ri] : 0, cnt = j < nkr ? c_mprstart[ri + 1] - first : 0;
      int inc = cnt;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
      }
      const int total = __shfl_sync(0xffffffffu, inc, 31), exc = inc - cnt;
      for (int t0 = 0; t0 < total; t0 += 32) {
        const int t = t0 + lane;
        int lo = 0;
#pragma unroll
        for (int st = 16; st; st >>= 1) {
          int e = __shfl_sync(0xffffffffu, exc, (lo + st) & 31);
          if (lo + st < 32 && e <= t) lo += st;
        }
        const int pr = __shfl_sync(0xffffffffu, first, lo) + t - __shfl_sync(0xffffffffu, exc, lo);
        const bool keep = t < total && pair_bound(s, pr, FN);
        const unsigned bal = __ballot_sync(0xffffffffu, keep);
        const int at = ncand + __popc(bal & lanes_below);
        if (keep && at < NCAND) s.cand[at] = (unsigned short)pr;
        ncand += __popc(bal);
      }
    }
#if NCAND < NPAIR
    if (ncand > NCAND) {  // candidates beyond capacity: their contacts are dropped
      if (lane == 0) s.unsupported |= 4;
      ncand = NCAND;
    }
#endif
#elif BROAD_GROUPS
"""

EDITS = (
    ("#if BROAD_GROUPS\n        struct { unsigned short cand[NCAND], gcand[NGRP]; };\n",
     "#if MIDPHASE\n        struct { MP_SCRATCH };\n#elif BROAD_GROUPS\n"
     "        struct { unsigned short cand[NCAND], gcand[NGRP]; };\n"),
    ("#if BROAD_GROUPS\n    struct { float gx[NGEOMC][3], gq[NGEOMC][4], con_dot[MAXCON][CD]; unsigned short cand[NCAND], gcand[NGRP]; };\n",
     "#if MIDPHASE\n    struct { float gx[NGEOMC][3], gq[NGEOMC][4], con_dot[MAXCON][CD]; MP_SCRATCH };\n#elif BROAD_GROUPS\n"
     "    struct { float gx[NGEOMC][3], gq[NGEOMC][4], con_dot[MAXCON][CD]; unsigned short cand[NCAND], gcand[NGRP]; };\n"),
    ("// world poses of the collision geoms (position, rotation quaternion) from the body poses\n",
     HELPER + "// world poses of the collision geoms (position, rotation quaternion) from the body poses\n"),
    ("    int ncand = 0;\n#if BROAD_GROUPS\n", "    int ncand = 0;\n" + DEVICE),
    ("#if HAS_DEFER\n  sort_contacts(s, lane);  // (the order of the split forward's, whose k_ccd appends in no fixed order)\n",
     "#if HAS_DEFER || MIDPHASE\n  sort_contacts(s, lane);  // (the order of the split forward's, whose k_ccd appends in no fixed "
     "order;\n  // the mid-phase: the lanes' order of arrival depends on the compiled code around collide_pair)\n"),
)


def apply(src, m):
    """The mid-phase's edits of the forest engine's text (adjoint.cuda_source, first) when enabled(m); else src."""
    if "#define MIDPHASE 1" not in src:
        return src
    for a, b in EDITS:
        assert src.count(a) == 1, f"midphase anchor found {src.count(a)} times: {a[:70]!r}"
        src = src.replace(a, b)
    return src
