"""Robot regression: each robot's forward against its MuJoCo Warp reference (robot_check.py check) and, with --grad N,
its gradients on N of those states against float64 central differences of MuJoCo C (robot_check.py grad64), one
process per robot, gated on recorded results.

Default: the committed fixtures, fixtures/robots (each robot's exported model, 512 reference states and the
licence; README.md there), gated on fixtures/robots/results.json, the results recorded on exactly those inputs.
--root DIR: another reference directory of the same layout, e.g. the full references behind
docs/results/robots_check_v100.json (not included), gated on that file (forward only)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "fixtures" / "robots"
BLOCKS = ("qpos", "qvel", "ctrl")


def _rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _run(a, env, cmd, model, ref, *extra):
    out = subprocess.run([a.python, "scripts/robots/robot_check.py", cmd, "--model", _rel(model), "--ref", _rel(ref),
                          "--api", a.api, *extra], capture_output=True, text=True, env=env, cwd=REPO)
    i = out.stdout.rfind("\n{\n")
    try:
        return json.loads(out.stdout[i + 1:] if i >= 0 else out.stdout)
    except ValueError:
        return dict(error=(out.stderr or out.stdout)[-400:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(FIXTURES))
    ap.add_argument("--robots", default=None, help="comma-separated (default: every recorded robot)")
    ap.add_argument("--api", choices=("torch", "jax"), default="jax", help="the engine interface of robot_check.py")
    ap.add_argument("--grad", type=int, default=0, help="worlds per robot of the gradient check (0: none)")
    ap.add_argument("--p99-factor", type=float, default=3.0)
    ap.add_argument("--p99-floor", type=float, default=5e-5)
    ap.add_argument("--ncon-slack", type=float, default=0.004)
    ap.add_argument("--grad-factor", type=float, default=3.0)
    ap.add_argument("--grad-floor", type=float, default=1e-5)
    ap.add_argument("--record", action="store_true", help="write the recorded results instead of gating")
    ap.add_argument("--merge", action="store_true", help="with --record: replace / add only these robots' entries")
    ap.add_argument("--python", default=sys.executable, help="the interpreter of the engine's environment")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    fixtures = (root / "results.json").exists() or a.record
    rec_file = root / "results.json" if fixtures else REPO / "docs" / "results" / "robots_check_v100.json"
    rec = {} if a.record and not rec_file.exists() else json.loads(rec_file.read_text())["robots"]
    names = a.robots.split(",") if a.robots else (sorted(p.name for p in root.iterdir() if (p / "ref.npz").exists())
                                                    if a.record else list(rec))
    pp = os.environ.get("PYTHONPATH")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(Path(x).resolve()) for x in pp.split(os.pathsep) if x)
               if pp else str(REPO / "src"))
    res, failed = {}, []
    for r in names:
        model, ref = root / r / f"{r}_flat", root / r / "ref.npz"
        got = dict(forward_vs_mjw=_run(a, env, "check", model, ref))
        if a.grad:
            got["gradient_vs_mujoco_c64"] = _run(a, env, "grad64", model, ref, "--nworld", str(a.grad))
        err = [v["error"] for v in got.values() if "error" in v]
        if err:
            res[r] = dict(error=err[0])
            failed.append(r)
            print(f"FAIL  {r}: no result ({err[0][-160:]!r})")
            continue
        res[r] = got
        if a.record:
            g = got.get("gradient_vs_mujoco_c64", {})
            print(f"REC   {r}: qvel p99 {got['forward_vs_mjw']['qvel_rel_p99']:.2e}, ncon equal "
                  f"{got['forward_vs_mjw']['ncon_equal']:.4f}" + "".join(
                      f", {b} {g[b]['agree_rel_median']:.1e}" for b in BLOCKS if b in g))
            continue
        f, wf = got["forward_vs_mjw"], rec[r]["forward_vs_mjw"]
        lim = max(a.p99_factor * wf["qvel_rel_p99"], a.p99_floor)
        why = []
        if not f["qvel_rel_p99"] <= lim:
            why.append(f"qvel p99 {f['qvel_rel_p99']:.2e} > {lim:.2e}")
        if not f["ncon_equal"] >= wf["ncon_equal"] - a.ncon_slack:
            why.append(f"ncon equal {f['ncon_equal']:.4f} < {wf['ncon_equal']:.4f} - {a.ncon_slack}")
        if f["flags"]:
            why.append(f"{f['flags']} flagged worlds")
        line = f"qvel p99 {f['qvel_rel_p99']:.2e} (recorded {wf['qvel_rel_p99']:.2e}), ncon equal {f['ncon_equal']:.4f}"
        if a.grad:
            g, wg = got["gradient_vs_mujoco_c64"], rec[r]["gradient_vs_mujoco_c64"]
            for b in BLOCKS:
                glim = max(a.grad_factor * wg[b]["agree_rel_median"], a.grad_floor)
                if not g[b]["agree_rel_median"] <= glim:
                    why.append(f"{b} gradient median {g[b]['agree_rel_median']:.2e} > {glim:.2e}")
            line += ", gradient medians " + " / ".join(f"{g[b]['agree_rel_median']:.1e}" for b in BLOCKS)
        res[r]["fails"] = why
        print(f"{'PASS' if not why else 'FAIL'}  {r}: {line}" + (f"; {'; '.join(why)}" if why else ""))
        if why:
            failed.append(r)
    if a.record:
        from nablaengine import provenance
        if failed:
            print(f"record: FAIL ({','.join(failed)}); nothing written")
            return 1
        prov = dict(commit=provenance.source_commit(), gpu=provenance.gpu(), api=a.api, grad=a.grad)
        if a.merge and rec_file.exists():
            out = json.loads(rec_file.read_text())
            for r, v in res.items():
                out["robots"][r] = dict(v, provenance=prov)
        else:
            out = dict(description="robot_check.py check (and grad64 on --grad worlds) on the fixtures' inputs; the "
                                   "recorded results of scripts/robots/regress.py", provenance=prov, robots=res)
        rec_file.write_text(json.dumps(out, indent=1) + "\n")
        print(f"record: {_rel(rec_file)}")
        return 0
    if a.out:
        Path(a.out).write_text(json.dumps(dict(results=res, failed=failed), indent=1) + "\n")
    print(f"regress: {'PASS' if not failed else 'FAIL ' + ','.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
