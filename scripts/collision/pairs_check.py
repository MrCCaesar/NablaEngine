"""Contacts of random geom pairs: MuJoCo Warp (any installed version) against the engine (the forest engine's collision,
one-substep jax_ffi calls), per pair type. For collision alignment work (MuJoCo Warp 3.17 vs 4.0, the engine vs both)."""
import argparse
import itertools
import json
import sys

import numpy as np

TYPES = ("plane", "sphere", "capsule", "box", "mesh")
CYL_PAIRS = ("plane-cylinder", "sphere-cylinder", "capsule-cylinder", "box-cylinder", "mesh-cylinder",
             "cylinder-cylinder")
ELL_PAIRS = ("plane-ellipsoid", "sphere-ellipsoid", "capsule-ellipsoid", "ellipsoid-ellipsoid", "ellipsoid-cylinder",
             "ellipsoid-box", "ellipsoid-mesh")
SIZES = {"sphere": "0.05", "capsule": "0.03 0.08", "box": "0.05 0.04 0.03", "cylinder": "0.04 0.06",
         "ellipsoid": "0.05 0.035 0.025"}
RB = {"sphere": 0.05, "capsule": 0.11, "box": 0.0707, "mesh": 0.05, "cylinder": 0.0721, "ellipsoid": 0.05}


def hull(seed):
    v = np.random.default_rng(seed).normal(size=(14, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True) * [0.05, 0.04, 0.035]


def geom(t, name, pair_only):
    flags = ' contype="0" conaffinity="0"' if pair_only else ""
    if t == "mesh":
        return f'<geom name="{name}" type="mesh" mesh="hull"{flags}/>'
    if t == "plane":
        return f'<geom name="{name}" type="plane" size="1 1 .1"{flags}/>'
    return f'<geom name="{name}" type="{t}" size="{SIZES[t]}"{flags}/>'


def model_xml(t1, t2, seed, nativeccd, pair_only):
    """Geom g1 (a plane on the world, or on a free body) and g2 on a free body; pair_only: as an explicit pair."""
    g1 = geom("plane", "g1", pair_only) if t1 == "plane" else f'<body name="b1"><freejoint/>{geom(t1, "g1", pair_only)}</body>'
    flag = "" if nativeccd else '<flag nativeccd="disable"/>'
    pair = '<contact><pair geom1="g1" geom2="g2"/></contact>' if pair_only else ""
    return (f'<mujoco><asset><mesh name="hull" vertex="{" ".join(map(str, hull(seed).ravel()))}"/></asset>'
            f'<option gravity="0 0 0">{flag}</option><worldbody>{g1}<body name="b2" pos="0 0 1"><freejoint/>'
            f'{geom(t2, "g2", pair_only)}</body></worldbody>{pair}</mujoco>')


def pair_types():
    return [f"{a}-{b}" for a, b in itertools.combinations_with_replacement(TYPES, 2) if b != "plane"] + list(CYL_PAIRS) + list(ELL_PAIRS)


def samples(name, n, seed):
    """(n, nq) qpos of the pair type's samples (its own random stream)."""
    t1, t2 = name.split("-")
    rng = np.random.default_rng([seed, pair_types().index(name)])

    def rquat():
        q = rng.normal(size=4)
        return q / np.linalg.norm(q)
    out = []
    for _ in range(n):
        if t1 == "plane":
            out.append(np.r_[0, 0, rng.uniform(-0.6, 1.0) * RB[t2], rquat()])
        else:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            out.append(np.r_[0, 0, 0, rquat(), (RB[t1] + RB[t2]) * rng.uniform(0.3, 0.95) * u, rquat()])
    return np.array(out)


def cmd_ref(a):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    out = {}
    for name in pair_types():
        t1, t2 = name.split("-")
        m = mujoco.MjModel.from_xml_string(model_xml(t1, t2, a.seed, not a.no_nativeccd, False))
        Q = samples(name, a.n, a.seed)
        mm = mjw.put_model(m)
        d = mjw.put_data(m, mujoco.MjData(m), nworld=a.n, nconmax=a.n * 8, njmax=a.n * 40)
        wp.copy(d.qpos, wp.array(Q.astype(np.float32)))
        mjw.forward(mm, d)
        nc = int(d.nacon.numpy()[0])
        wid, dist = d.contact.worldid.numpy()[:nc], d.contact.dist.numpy()[:nc]
        pos, fr = d.contact.pos.numpy()[:nc], d.contact.frame.numpy()[:nc]
        out[name] = np.array([np.array([(dist[k], *pos[k], *fr[k][0]) for k in range(nc) if wid[k] == w]).reshape(-1, 7)
                              for w in range(a.n)], dtype=object)
    meta = dict(source="mujoco_warp", mujoco=mujoco.__version__, warp=wp.__version__, n=a.n, seed=a.seed,
                nativeccd=not a.no_nativeccd)
    np.savez(a.out, meta=np.array(json.dumps(meta)), **out, allow_pickle=True)
    print(json.dumps(meta))


def cmd_engine(a):
    import jax.numpy as jnp
    import mujoco
    from nablaengine import jax_ffi
    t1, t2 = a.pair.split("-")
    m = mujoco.MjModel.from_xml_string(model_xml(t1, t2, a.seed, not a.no_nativeccd, True))
    Q = samples(a.pair, a.n, a.seed)
    from nablaengine import engines
    engines.prefer_forest(m)
    jax_ffi.init(model=m, foot_pairs=[])
    z = jnp.zeros((a.n, m.nv))
    out = jax_ffi.step(jnp.asarray(Q, jnp.float32), z, z, jnp.zeros((a.n, m.nu)), jnp.ones(a.n), jnp.zeros((a.n, 3)),
                       substeps=1, iterations=1, contacts=True)
    c = {k: np.asarray(v) for k, v in jax_ffi.contact_table(out[3]).items()}
    rows = [np.array([(c["dist"][w, k], *c["pos"][w, k], *c["normal"][w, k]) for k in range(c["valid"].shape[1])
                      if c["valid"][w, k]]).reshape(-1, 7) for w in range(a.n)]
    import os
    meta = dict(source="nablaengine", n=a.n, seed=a.seed, nativeccd=not a.no_nativeccd,
                switches={k: os.environ[k] for k in ("DK_GJK_MJ4",) if os.environ.get(k)})
    np.savez(a.out, meta=np.array(json.dumps(meta)), **{a.pair: np.array(rows, dtype=object)}, allow_pickle=True)
    print(json.dumps(meta), sum(len(r) for r in rows), "contacts")


def stats(A, B):
    cnt, errs = 0, []
    for x, y in zip(A, B):
        if len(x) != len(y):
            cnt += 1
            continue
        if not len(x):
            continue
        used, e = set(), 0.0
        for r in x:
            j = min((j for j in range(len(y)) if j not in used), key=lambda j: np.abs(y[j][1:4] - r[1:4]).sum())
            used.add(j)
            e = max(e, float(np.abs(y[j] - r).max()))
        errs.append(e)
    e = np.array(errs or [0.0])
    return dict(count_differs=cnt, above_1e_4=int((e > 1e-4).sum()), p99=float(np.percentile(e, 99)), max=float(e.max()))


def cmd_compare(a):
    files = [np.load(f, allow_pickle=True) for f in a.files]
    labels = [json.loads(str(f["meta"])) for f in files]
    base = files[0]
    print("reference:", labels[0])
    for f, lab in zip(files[1:], labels[1:]):
        for name in f.files:
            if name == "meta" or name not in base.files:
                continue
            print(f"{name:14s} {lab.get('source')} {lab.get('mujoco', '')} {lab.get('switches', '')}: {stats(f[name], base[name])}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ref", "engine"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--n", type=int, default=400)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--no-nativeccd", action="store_true")
        if name == "engine":
            p.add_argument("--pair", required=True, choices=pair_types())
    p = sub.add_parser("compare")
    p.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"ref": cmd_ref, "engine": cmd_engine, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
