"""The BAM servo model's adjoint on the tree engine (init(actuator=Bam(...))) against central differences: the float64
MuJoCo C reference (nablaengine.actuators.bam.Reference, converged solves) and the engine's own float32 forward."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nablaengine import provenance
from nablaengine.actuators import Bam, bam
import bam_check

GATES = {"g1": dict(one_med=1e-3, one_p90=1e-2, roll_med=1e-2, f32_med=2e-2),
         # microduck (bam_check.py's docstring): the G1's gates until a CUDA run sets them (the Mac port's Metal build
         # without the current limit, 32 states: medians 3e-7 .. 5e-6 against the same float64 central differences)
         "microduck": dict(one_med=1e-3, one_p90=1e-2, roll_med=1e-2, f32_med=2e-2)}


def groups(m):
    """{name: (array, index list)}: the input coordinates checked (array: qpos, qvel, ctrl, act)."""
    nv, nq, nu = m.nv, m.nq, m.nu
    return {"qpos_root": ("qpos", list(range(3))), "qpos_hinge": ("qpos", list(range(7, nq))),
            "qvel": ("qvel", list(range(nv))), "ctrl": ("ctrl", list(range(nu))),
            "tau": ("act", list(range(6, nv))), "ext": ("act", list(range(nv + 6, 2 * nv))),
            "vin0": ("act", [2 * nv]), "drop": ("act", [2 * nv + 1])}


class Engine:
    def __init__(self, backend, m, fp, com, b):
        self.backend = backend
        if backend == "jax":
            import jax
            import jax.numpy as jnp
            from nablaengine import jax_ffi
            self.info = jax_ffi.init(model=m, foot_pairs=fp, com_body=com, actuator=b)
            self.mod, self.jax = jax_ffi, jax
            self.cv = lambda x: jnp.asarray(np.ascontiguousarray(x, np.float32))
        else:
            import torch
            from nablaengine import torch_ffi
            self.info = torch_ffi.init(model=m, foot_pairs=fp, com_body=com, actuator=b)
            self.mod, self.torch = torch_ffi, torch
            self.cv = lambda x: torch.as_tensor(np.ascontiguousarray(x, np.float32), device="cuda")

    @staticmethod
    def np(x):
        return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)

    def forward(self, q, v, w, u, f, a, it, sub, tape=False):
        B = q.shape[0]
        r = self.mod.forward(*map(self.cv, (q, v, w, u, f, np.zeros((B, 3)))), act=self.cv(a), iterations=it,
                             substeps=sub, tape=tape)
        return [self.np(x) for x in r]

    def backward(self, tape, f, a, cq, cv, ca, sub):
        B = tape.shape[0]
        r = self.mod.backward(*map(self.cv, (tape, f, np.zeros((B, 3)), cq, cv)), substeps=sub, clamp_kv=1, normal=0,
                              act=self.cv(a), g_act=self.cv(ca))
        return [self.np(x) for x in r]

    def step_outputs(self, q, v, w, u, f, a, it, sub):
        """The differentiable step's primal outputs (q', v', w', act')."""
        B = q.shape[0]
        args = list(map(self.cv, (q, v, w, u, f, np.zeros((B, 3)), a)))
        if self.backend == "jax":
            fn = self.mod.make_step(iterations=it, substeps=sub, clamp_kv=1, normal=0)
            out, _ = self.jax.vjp(fn, *args)
        else:
            for x in (args[0], args[1], args[3], args[6]):
                x.requires_grad_(True)
            out = self.mod.step(*args[:6], act=args[6], iterations=it, substeps=sub, clamp_kv=1, normal=0)
        return [self.np(x) for x in out[:4]]

    def rollout_grad(self, q0, v0, w0, U, f, a0, cq, cv, ca, it, sub):
        """d L(final state) / d (q0, v0, a0, U) through len(U) control steps (L = cq.q + cv.v + ca.act)."""
        B = q0.shape[0]
        com = np.zeros((B, 3))
        if self.backend == "jax":
            jnp = self.jax.numpy
            fn = self.mod.make_step(iterations=it, substeps=sub, clamp_kv=1, normal=0)
            fr, cm, w = self.cv(f), self.cv(com), self.cv(w0)

            def loss(q, v, a, U):
                ww = w
                for k in range(U.shape[0]):
                    q, v, ww, a = fn(q, v, ww, U[k], fr, cm, a)
                return jnp.sum(q * cq) + jnp.sum(v * cv) + jnp.sum(a * ca)

            g = self.jax.grad(loss, argnums=(0, 1, 2, 3))(self.cv(q0), self.cv(v0), self.cv(a0), self.cv(U))
            return [self.np(x) for x in g]
        torch = self.torch
        q, v, a, Ut = (self.cv(x).requires_grad_(True) for x in (q0, v0, a0, U))
        qq, vv, aa, ww = q, v, a, self.cv(w0)
        fr, cm = self.cv(f), self.cv(com)
        for k in range(U.shape[0]):
            qq, vv, ww, aa = self.mod.step(qq, vv, ww, Ut[k], fr, cm, act=aa, iterations=it, substeps=sub, clamp_kv=1,
                                           normal=0)
        L = (qq * self.cv(cq)).sum() + (vv * self.cv(cv)).sum() + (aa * self.cv(ca)).sum()
        L.backward()
        return [self.np(x.grad) for x in (q, v, a, Ut)]


def stats(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(x.max()))


def compare(G, FD, keep, gnames, sel=None):
    """Per input group: the per-world relative error of the gradient G[name] (n, k) against FD[name] over the kept
    coordinates, the entries' errors relative to their world's ||fd||, the excluded count."""
    out = {}
    for name in gnames:
        g, fd, kp = G[name], FD[name], keep[name]
        rows = range(g.shape[0]) if sel is None else np.flatnonzero(sel)
        rel, ent, scale_all = [], [], max(float(np.abs(fd[kp]).max()) if kp.any() else 0.0, 1e-30)
        for i in rows:
            k = kp[i]
            if not k.any():
                continue
            nf = np.linalg.norm(fd[i, k])
            den = max(nf, 1e-6 * scale_all)
            rel.append(np.linalg.norm(g[i, k] - fd[i, k]) / den)
            ent.extend((np.abs(g[i, k] - fd[i, k]) / den).tolist())
        out[name] = dict(world_rel=stats(rel), entry_rel=stats(ent), excluded=int((~kp[list(rows)]).sum()),
                         coords=int(kp[list(rows)].size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="g1", choices=tuple(GATES))
    ap.add_argument("--backend", default="jax", choices=("jax", "torch"))
    ap.add_argument("--max-current", default="firmware")
    ap.add_argument("--worlds", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--rollout-worlds", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--eps32", type=float, default=1e-3)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--lift", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--dump")
    a = ap.parse_args()
    mc = {"firmware": "firmware", "none": None}.get(a.max_current, None)
    if a.max_current not in ("firmware", "none"):
        mc = float(a.max_current)
    b = Bam(max_current=mc)
    m0, fp, com, q0, sub = bam_check.load(a.model)
    if a.model == "g1":
        a.lift = 3.0 if a.lift is None else a.lift
    q0[2] += a.lift or 0.0
    m0.opt.iterations, m0.opt.ls_iterations = a.iterations, a.iterations
    ref = bam.Reference(m0, b)
    ref.m.opt.tolerance, ref.m.opt.ls_tolerance = 0.0, 0.0
    m, sv = ref.m, bam.servo(b)
    nq, nv, nu = m.nq, m.nv, m.nu
    t0 = time.perf_counter()
    eng = Engine(a.backend, m0, fp, com, b)
    t_init = time.perf_counter() - t0
    it, rng = a.iterations, np.random.default_rng(a.seed)
    G = groups(m)

    Z = bam_check.corpus(ref, q0, sub, a.worlds, rng)
    n = len(Z["qpos"])
    fric = np.full(n, m.pair_friction[fp[0], 0])
    cq = rng.normal(0, 1, (n, nq)); cq[:, 3:7] = 0.0
    cv = rng.normal(0, 1, (n, nv))
    ca = np.zeros((n, 2 * nv + 2)); ca[:, 6:nv] = rng.normal(0, 1, (n, nu)); ca[:, nv + 6:2 * nv] = rng.normal(0, 1, (n, nu))
    ca[:, 2 * nv:] = rng.normal(0, 1, (n, 2))
    cq, cv, ca = (np.float32(x).astype(float) for x in (cq, cv, ca))
    X = {k: Z[k] for k in ("qpos", "qvel", "warm", "ctrl", "act")}

    r0 = eng.forward(X["qpos"], X["qvel"], X["warm"], X["ctrl"], fric, X["act"], it, sub)
    r1 = eng.forward(X["qpos"], X["qvel"], X["warm"], X["ctrl"], fric, X["act"], it, sub, tape=True)
    rs = eng.step_outputs(X["qpos"], X["qvel"], X["warm"], X["ctrl"], fric, X["act"], it, sub)
    tape_ok = all(np.array_equal(x, y) for x, y in zip(r0[:5], r1[:5])) and \
        all(np.array_equal(x, y) for x, y in zip(r0[:4], rs))
    tp = r1[-1]
    rec = int(eng.info["rec_words"])
    off = rec - bam.tape_words(nv)
    flags = tp.reshape(n, sub, rec)[:, :, off + 1 + 3 * nv + 6: off + 1 + 4 * nv]
    cur_world = (flags != 0).any(axis=(1, 2))
    act_rec = tp.reshape(n, sub, rec)[:, :, off + 1 + 2 * nv + 6: off + 1 + 3 * nv]
    scale = tp.reshape(n, sub, rec)[:, :, off]
    volt = np.abs(np.abs(act_rec) - scale[:, :, None] * sv["frclim"]) <= 1e-6 * sv["frclim"]
    lim_world = cur_world | volt.any(axis=(1, 2))
    limits = dict(current_limited_dof_substeps=int((flags != 0).sum()), worlds_with_current_limit=int(cur_world.sum()),
                  voltage_limited_dof_substeps=int(volt.sum()), worlds_with_voltage_limit=int(volt.any(axis=(1, 2)).sum()),
                  dof_substeps=int(flags.size))

    gq, gv, gu, ga, ginfo = eng.backward(tp, fric, X["act"], cq, cv, ca, sub)
    GE = {"qpos_root": gq[:, :3], "qpos_hinge": gq[:, 7:], "qvel": gv, "ctrl": gu, "tau": ga[:, 6:nv],
          "ext": ga[:, nv + 6:2 * nv], "vin0": ga[:, 2 * nv:2 * nv + 1], "drop": ga[:, 2 * nv + 1:]}

    def L64(i, X1):
        q1, v1, _, a1 = ref.step(X1["qpos"], X1["qvel"], X1["warm"], X1["ctrl"], X1["act"], sub)
        return float(cq[i] @ q1 + cv[i] @ v1 + ca[i] @ a1)

    FD, KEEP = {}, {}
    t1 = time.perf_counter()
    for name, (arr, idx) in G.items():
        FD[name], KEEP[name] = np.zeros((n, len(idx))), np.zeros((n, len(idx)), bool)
        for i in range(n):
            base = {k: X[k][i].astype(float).copy() for k in X}
            for j, c in enumerate(idx):
                d = []
                for e in (a.eps, a.eps / 4):
                    xp, xm = dict(base), dict(base)
                    xp[arr], xm[arr] = base[arr].copy(), base[arr].copy()
                    xp[arr][c] += e
                    xm[arr][c] -= e
                    d.append((L64(i, xp) - L64(i, xm)) / (2 * e))
                FD[name][i, j] = d[1]
                KEEP[name][i, j] = abs(d[0] - d[1]) <= 1e-4 * max(abs(d[1]), 1e-2)
    t_fd = time.perf_counter() - t1
    one = compare(GE, FD, KEEP, G)
    one_cur = compare(GE, FD, KEEP, G, lim_world)
    one_free = compare(GE, FD, KEEP, G, ~lim_world)

    FD32, KEEP32 = {}, {}
    for name, (arr, idx) in G.items():
        FD32[name] = np.zeros((n, len(idx)))
        KEEP32[name] = np.zeros((n, len(idx)), bool)
        for j, c in enumerate(idx):
            e0 = a.eps32 * {"qpos": 0.1, "vin0": 10.0}.get(arr if arr == "qpos" else name, 1.0)
            d = []
            for e in (e0, 2 * e0):
                Lpm = []
                for sgn in (1, -1):
                    Y = {k: X[k].copy() for k in X}
                    Y[arr] = Y[arr].copy()
                    Y[arr][:, c] += sgn * e
                    o = eng.forward(Y["qpos"], Y["qvel"], Y["warm"], Y["ctrl"], fric, Y["act"], it, sub)
                    Lpm.append((o[0] * cq).sum(1) + (o[1] * cv).sum(1) + (o[3] * ca).sum(1))
                d.append((Lpm[0] - Lpm[1]) / (2 * e))
            FD32[name][:, j] = d[0]
            KEEP32[name][:, j] = np.abs(d[0] - d[1]) <= 0.05 * np.maximum(np.abs(d[0]), 1e-2)
    own = compare(GE, FD32, KEEP32, G)

    R, H = min(a.rollout_worlds, n), a.horizon
    lw, fw = np.flatnonzero(lim_world), np.flatnonzero(~lim_world)
    nl = min(len(lw), R // 2)
    sel = np.concatenate([lw[:nl], fw[:R - nl]]) if len(fw) >= R - nl else np.arange(R)
    X = {k: v[sel] for k, v in X.items()}
    cq, cv, ca, fric = cq[sel], cv[sel], ca[sel], fric[sel]
    U = X["ctrl"][None] + rng.normal(0, 0.15, (H, R, nu))
    U = np.float32(U).astype(float)
    gr = eng.rollout_grad(X["qpos"], X["qvel"], X["warm"], U, fric, X["act"], cq, cv, ca, it, sub)
    RE = {"qpos_root": gr[0][:, :3], "qpos_hinge": gr[0][:, 7:], "qvel": gr[1], "tau": gr[2][:, 6:nv],
          "ext": gr[2][:, nv + 6:2 * nv], "vin0": gr[2][:, 2 * nv:2 * nv + 1], "drop": gr[2][:, 2 * nv + 1:],
          "ctrl_first": gr[3][0], "ctrl_last": gr[3][-1], "ctrl_all": gr[3].transpose(1, 0, 2).reshape(R, H * nu)}

    def roll64(i, q, v, w, Ui, act):
        for k in range(H):
            q, v, w, act = ref.step(q, v, w, Ui[k], act, sub)
        return float(cq[i] @ q + cv[i] @ v + ca[i] @ act)

    RG = dict(G)
    RG.pop("ctrl")
    RG["ctrl_all"] = ("U", list(range(H * nu)))
    RFD, RKEEP = {}, {}
    t2 = time.perf_counter()
    for name, (arr, idx) in RG.items():
        RFD[name], RKEEP[name] = np.zeros((R, len(idx))), np.zeros((R, len(idx)), bool)
        for i in range(R):
            base = dict(qpos=X["qpos"][i].astype(float), qvel=X["qvel"][i].astype(float), act=X["act"][i].astype(float),
                        U=U[:, i].reshape(-1).copy())
            for j, c in enumerate(idx):
                d = []
                for e in (a.eps, a.eps / 4):
                    vals = []
                    for sgn in (1, -1):
                        y = {k: v.copy() for k, v in base.items()}
                        y[arr][c] += sgn * e
                        vals.append(roll64(i, y["qpos"], y["qvel"], X["warm"][i].astype(float), y["U"].reshape(H, nu),
                                           y["act"]))
                    d.append((vals[0] - vals[1]) / (2 * e))
                RFD[name][i, j] = d[1]
                RKEEP[name][i, j] = abs(d[0] - d[1]) <= 1e-4 * max(abs(d[1]), 1e-2)
    t_rfd = time.perf_counter() - t2
    RFD["ctrl_first"], RKEEP["ctrl_first"] = RFD["ctrl_all"][:, :nu], RKEEP["ctrl_all"][:, :nu]
    RFD["ctrl_last"], RKEEP["ctrl_last"] = RFD["ctrl_all"][:, -nu:], RKEEP["ctrl_all"][:, -nu:]
    roll = compare(RE, RFD, RKEEP, list(RG) + ["ctrl_first", "ctrl_last"])

    g = GATES[a.model]
    rows = []
    for name in G:
        s = one[name]["world_rel"]
        if s.get("n"):
            rows.append((f"one step {name} median", s["median"], g["one_med"]))
            rows.append((f"one step {name} p90", s["p90"], g["one_p90"]))
        s = own[name]["world_rel"]
        if s.get("n"):
            rows.append((f"own float32 fd {name} median", s["median"], g["f32_med"]))
    for name in roll:
        s = roll[name]["world_rel"]
        if s.get("n"):
            rows.append((f"rollout {name} median", s["median"], g["roll_med"]))
    ok = tape_ok and all(v <= t for _, v, t in rows) and int((ginfo != 0).sum()) == 0
    res = dict(check="bam_grad_check", model=a.model, backend=a.backend,
               engine=eng.info.get("engine"), bam=b.key(), substeps=sub, iterations=it,
               lift=a.lift or 0.0, seed=a.seed, worlds=n, eps=a.eps, eps32=a.eps32, horizon=H, rollout_worlds=R,
               rollout_worlds_with_limits=int(nl),
               init_seconds=t_init, fd_seconds=dict(one=t_fd, rollout=t_rfd), tape_bitwise=tape_ok, limits=limits,
               adjoint_info_nonzero=int((ginfo != 0).sum()), one_step=one, one_step_limited_worlds=one_cur,
               one_step_other_worlds=one_free, own_float32_fd=own, rollout=roll,
               gates={k: dict(value=v, limit=t, ok=v <= t) for k, v, t in rows}, ok=ok,
               provenance=provenance.provenance())

    def line(tag, d):
        for name, x in d.items():
            s, e = x["world_rel"], x["entry_rel"]
            if s.get("n"):
                print(f"{tag:10s} {name:11s} world rel med {s['median']:.1e} p90 {s['p90']:.1e} max {s['max']:.1e} | "
                      f"entry med {e['median']:.1e} p90 {e['p90']:.1e} max {e['max']:.1e} | excl {x['excluded']}/{x['coords']}")
    print(json.dumps(dict(model=a.model, backend=a.backend, bam=b.key(), tape_bitwise=tape_ok, limits=limits)))
    line("one", one)
    line("one/lim", one_cur)
    line("own f32", own)
    line("rollout", roll)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    if a.dump:
        np.savez(a.dump, gq=gq, gv=gv, gu=gu, ga=ga, **{"roll%d" % k: x for k, x in enumerate(gr)})
    bad = [k for k, v, t in rows if v > t]
    print("PASS" if ok else f"FAIL {bad}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
