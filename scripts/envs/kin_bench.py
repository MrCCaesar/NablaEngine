"""Timing of envs.kin's forward kinematics, batched form vs the per-body reference (scripts/envs/kin_check.py), per
model at --batch worlds: ms per fk forward (qpos, qvel -> Frames) and per backward (the gradient of a random linear
function of xpos, xmat, w, v0 w.r.t. qpos and qvel), for"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kin_check import fk_reference, random_states


def _interleaved_ms(cases, reps, sync):
    ts = {kk: [] for kk in cases}
    for _ in range(reps):
        for kk, (setup, run) in cases.items():
            setup()
            sync()
            t = time.perf_counter()
            run()
            sync()
            ts[kk].append(time.perf_counter() - t)
    return {kk: 1e3 * float(np.median(v)) for kk, v in ts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="cheetah-run,humanoid-run,walker-run,brax-ant")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modes", default="torch,graph,jax")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (CPU)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    modes = a.modes.split(",")

    import mujoco
    from nablaengine import envs
    from nablaengine.envs import kin as kin_mod
    nothing = lambda: None
    report = {"batch": a.batch, "device": a.device, "threads": a.threads, "tasks": {}}
    for name in a.tasks.split(","):
        xml, assets = envs.spec(name).source()
        m = mujoco.MjModel.from_xml_string(xml, assets)
        k = kin_mod.Kinematics(m)
        qpos, qvel = random_states(m, a.batch, np.random.default_rng(0))
        wts = {n: np.random.default_rng(1).normal(0, 1, s) for n, s in
               (("xpos", (m.nbody, 3)), ("xmat", (m.nbody, 3, 3)), ("w", (m.nbody, 3)), ("v0", (m.nbody, 3)))}
        fks = {"new": lambda o, q, v: k.fk(o, q, v), "ref": lambda o, q, v: fk_reference(k, o, q, v)}
        r = {}
        if "torch" in modes or "graph" in modes:
            import torch
            from torch.profiler import profile, ProfilerActivity
            from nablaengine.envs.ops import TorchOps
            if a.threads:
                torch.set_num_threads(a.threads)
            o = TorchOps(a.device)
            cuda = o.device.type == "cuda"
            sync = torch.cuda.synchronize if cuda else nothing
            W = {n: torch.as_tensor(x, dtype=torch.float32, device=o.device) for n, x in wts.items()}

            def leaves():
                return (torch.as_tensor(qpos, dtype=torch.float32, device=o.device).requires_grad_(True),
                        torch.as_tensor(qvel, dtype=torch.float32, device=o.device).requires_grad_(True))

            def loss(fk, q, v):
                F = fk(o, q, v)
                return sum((getattr(F, n) * W[n]).sum() for n in W)

            def nograd(fk, q, v):
                with torch.no_grad():
                    fk(o, q, v)
            if "torch" in modes:
                q, v = leaves()
                cases, held = {}, {}
                for tag, fk in fks.items():
                    for _ in range(5):
                        nograd(fk, q, v)
                        torch.autograd.grad(loss(fk, q, v), (q, v))

                    def setup_b(tag=tag, fk=fk):
                        held[tag] = loss(fk, q, v)
                    cases[(tag, "fwd")] = (nothing, lambda fk=fk: nograd(fk, q, v))
                    cases[(tag, "bwd")] = (setup_b, lambda tag=tag: torch.autograd.grad(held.pop(tag), (q, v)))
                t = _interleaved_ms(cases, a.reps, sync)
                acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])

                def counts(p):
                    ev = p.events()
                    out = [len([e for e in ev if e.name.startswith("aten::")])]
                    if cuda:
                        kern = [e for e in ev if e.device_type.name == "CUDA"]
                        out += [len(kern), 1e-3 * sum(e.device_time for e in kern)]
                    return out
                for tag, fk in fks.items():
                    with profile(activities=acts) as p:
                        nograd(fk, q, v)
                        sync()
                    cf = counts(p)
                    L = loss(fk, q, v)
                    with profile(activities=acts) as p:
                        torch.autograd.grad(L, (q, v))
                        sync()
                    cb = counts(p)
                    del L
                    e = {"fwd_ms": t[(tag, "fwd")], "bwd_ms": t[(tag, "bwd")],
                         "fwd_aten_ops": cf[0], "bwd_aten_ops": cb[0]}
                    if cuda:
                        e.update(fwd_kernels=cf[1], bwd_kernels=cb[1], fwd_kernel_ms=cf[2], bwd_kernel_ms=cb[2])
                    r.setdefault("torch", {})[tag] = e
        if "graph" in modes and cuda:
                cases = {}
                for tag, fk in fks.items():
                    q, v = leaves()
                    st = torch.cuda.Stream()
                    st.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(st):
                        for _ in range(3):
                            nograd(fk, q, v)
                            torch.autograd.grad(loss(fk, q, v), (q, v))
                    torch.cuda.current_stream().wait_stream(st)
                    gf, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gf):
                        nograd(fk, q, v)
                    with torch.cuda.graph(gb):
                        torch.autograd.grad(loss(fk, q, v), (q, v))
                    cases[(tag, "fwd")] = (nothing, gf.replay)
                    cases[(tag, "fwd_bwd")] = (nothing, gb.replay)
                t = _interleaved_ms(cases, a.reps, sync)
                for tag in fks:
                    r.setdefault("graph", {})[tag] = {"fwd_ms": t[(tag, "fwd")], "fwd_bwd_ms": t[(tag, "fwd_bwd")],
                                                      "bwd_ms": t[(tag, "fwd_bwd")] - t[(tag, "fwd")]}
        if "jax" in modes:
            import jax
            import jax.numpy as jnp
            from nablaengine.envs.ops import JaxOps
            oj = JaxOps()
            qj, vj = jnp.asarray(qpos, jnp.float32), jnp.asarray(qvel, jnp.float32)
            Wj = {n: jnp.asarray(x, jnp.float32) for n, x in wts.items()}
            cases, hlo = {}, {}
            for tag, fk in fks.items():
                def f(q, v, fk=fk):
                    F = fk(oj, q, v)
                    return F.xpos, F.xmat, F.w, F.v0

                def lossj(q, v, fk=fk):
                    F = fk(oj, q, v)
                    return sum((getattr(F, n) * Wj[n]).sum() for n in Wj)
                jf = jax.jit(f)
                jg = jax.jit(jax.grad(lossj, argnums=(0, 1)))
                for _ in range(3):
                    jax.block_until_ready((jf(qj, vj), jg(qj, vj)))
                hlo[tag] = len(jg.lower(qj, vj).compile().as_text().splitlines())
                cases[(tag, "fwd")] = (nothing, lambda g=jf: jax.block_until_ready([g(qj, vj) for _ in range(10)]))
                cases[(tag, "fwd_bwd")] = (nothing, lambda g=jg: jax.block_until_ready([g(qj, vj) for _ in range(10)]))
            t = {kk: x / 10 for kk, x in _interleaved_ms(cases, a.reps, nothing).items()}
            for tag in fks:
                r.setdefault("jax", {})[tag] = {"fwd_ms": t[(tag, "fwd")], "fwd_bwd_ms": t[(tag, "fwd_bwd")],
                                                "bwd_ms": t[(tag, "fwd_bwd")] - t[(tag, "fwd")],
                                                "grad_hlo_lines": hlo[tag], "device": str(jax.devices()[0])}
        report["tasks"][name] = r
        print(name, json.dumps(r), file=sys.stderr, flush=True)
    s = json.dumps(report, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
