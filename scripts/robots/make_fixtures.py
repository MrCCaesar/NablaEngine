"""Make the committed robot fixtures (fixtures/robots) from the full exports and references under --root (each robot's
export_robot.py model and its robot_check.py ref file): the model's XML as exported, its JSON with the source named
as in docs/results/robots_check_v100.json (no local paths) and today's export_robot.py option names, and --nworld
reference states drawn at random (seed 0, in their original order). The licences are copied by hand
(fixtures/robots/README.md). A robot outside robots_check_v100.json (fixtures added later, e.g. MuJoCo Menagerie's
anymal_b) takes its source from --source."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "runs" / "robots"))
    ap.add_argument("--out", default=str(REPO / "fixtures" / "robots"))
    ap.add_argument("--robots", default=None, help="comma-separated (default: every robot of the recorded results)")
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--source", default=None, help="the source of a robot outside robots_check_v100.json (one robot)")
    a = ap.parse_args()
    rec = json.loads((REPO / "docs" / "results" / "robots_check_v100.json").read_text())["robots"]
    robots = a.robots.split(",") if a.robots else list(rec)
    for r in robots:
        if r not in rec:
            if not a.source or len(robots) != 1:
                raise SystemExit(f"{r}: not in robots_check_v100.json; give its --source (one robot per run)")
            rec[r] = dict(source=a.source)
        src, dst = Path(a.root) / r, Path(a.out) / r
        dst.mkdir(parents=True, exist_ok=True)
        name = rec[r].get("model", f"{r}_flat")
        shutil.copyfile(src / f"{name}.xml", dst / f"{r}_flat.xml")
        if (src / "meshes").is_dir():
            shutil.copytree(src / "meshes", dst / "meshes", dirs_exist_ok=True)
        meta = json.loads((src / f"{name}.json").read_text())
        meta["source"] = rec[r]["source"]
        if "training" in meta:
            meta = {("locomotion" if k == "training" else k): v for k, v in meta.items()}
        (dst / f"{r}_flat.json").write_text(json.dumps(meta, indent=1) + "\n")
        z = np.load(src / rec[r].get("ref", "ref.npz"))
        n = len(z["qpos"])
        idx = np.sort(np.random.default_rng(0).choice(n, min(a.nworld, n), replace=False))
        out = {k: (z[k] if z[k].ndim == 0 else z[k][idx]) for k in z.files}
        np.savez_compressed(dst / "ref.npz", **out, source_rows=idx)
        print(f"{r}: {len(idx)} of {n} states, {(dst / 'ref.npz').stat().st_size} bytes")


if __name__ == "__main__":
    main()
