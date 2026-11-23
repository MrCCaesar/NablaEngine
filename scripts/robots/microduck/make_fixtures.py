"""fixtures/robots/microduck/ from pollen-robotics/microduck_rl, every step after the scenes:

  1. (before this script, with MuJoCo 3.10.0, mjlab 1.3.0's pin: README.md) build_scenes.py writes model/scene_spheres4.xml:
     the walk model with the Velocity task's options and 4 spheres per foot (fit_results.json, from fit_feet.py).
     MuJoCo 3.17 writes this scene with inertial frames that differ in the fifth digit, so the committed fixture is tied
     to the scene written by 3.10.0;
  2. scripts/robots/export_robot.py --mjcf <scene> --name microduck --foot-geoms <the spheres> --com-body trunk_base
     --feet-only --key STAND: the engine model (explicit floor-foot pairs, visual geoms dropped);
  3. here: the three collision geoms that are in no contact pair, and their meshes, removed (the robot's meshes are
     CC BY-NC-SA 4.0 and not distributed; masses, inertias and the pairs do not change), and the json's provenance.

The servo values of the BAM model are not part of the fixture: init(actuator="bam") prepares the model
(nablaengine.actuators.bam.prepare).

usage: python make_fixtures.py <microduck_rl checkout> [--scenes DIR] [--check]    (python with mujoco 3.17.0, numpy)
--scenes: the directory build_scenes.py wrote into (default: model/ beside this script).
--check: compare with the committed files instead of writing them (exit status 1 on a difference).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "fixtures" / "robots" / "microduck"
COMMIT = "bae39f5"
NAME, SCENE = "microduck_spheres4", "scene_spheres4.xml"
FEET = [f"{s}_foot_s{i}_collision" for s in ("left", "right") for i in range(4)]
DROP = ("trunk_base_collision0", "leg_collision0", "leg_2_collision0")


def drop_unpaired(base):
    spec = mujoco.MjSpec.from_file(str(base) + ".xml")
    for g in DROP:
        spec.delete(spec.geom(g))
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for me in list(spec.meshes):
        if me.name not in used:
            spec.delete(me)
    Path(str(base) + ".xml").write_text(spec.to_xml())


def main():
    argv = sys.argv[1:]
    check = "--check" in argv
    scenes = Path(argv[argv.index("--scenes") + 1]) if "--scenes" in argv else HERE / "model"
    args = [a for i, a in enumerate(argv) if a not in ("--check", "--scenes") and (i == 0 or argv[i - 1] != "--scenes")]
    if len(args) != 1:
        sys.exit(__doc__)
    src = Path(args[0]) / "src" / "mjlab_microduck" / "robot" / "microduck"
    head = subprocess.run(["git", "-C", args[0], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not head.startswith(COMMIT):
        print(f"warning: the checkout is at {head[:7]}, the fixture was made from {COMMIT}")
    if not (scenes / SCENE).exists():
        sys.exit(f"{scenes / SCENE} not found: run build_scenes.py first (README.md)")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    with tempfile.TemporaryDirectory(prefix="microduck_fixtures_") as tmp:
        new = Path(tmp)
        export = ["--mjcf", f"<scenes>/{SCENE}", "--name", "microduck", "--foot-geoms", ",".join(FEET), "--com-body", "trunk_base",
                  "--feet-only", "--key", "STAND"]
        cmd = [sys.executable, str(REPO / "scripts" / "robots" / "export_robot.py")] + [a.replace("<scenes>", str(scenes)) for a in export]
        subprocess.run(cmd + ["--out", str(new / NAME)], check=True, env=env, capture_output=True)
        drop_unpaired(new / NAME)
        meta = json.loads((new / f"{NAME}.json").read_text())
        meta["source"] = (f"pollen-robotics/microduck_rl @ {COMMIT}, src/mjlab_microduck/robot/microduck/scene_walk.xml; "
                          "the sole meshes replaced by 4 spheres per foot; scripts/robots/microduck/make_fixtures.py")
        meta["provenance"] = dict(
            microduck_rl=COMMIT, sha256={f: sha(src / f) for f in ("scene_walk.xml", "robot_walk.xml")},
            scene=f"scripts/robots/microduck/build_scenes.py (MuJoCo 3.10.0) -> {SCENE}, sha256 {sha(scenes / SCENE)}",
            export="scripts/robots/export_robot.py " + " ".join(export),
            removed="collision geoms without a contact pair, and their meshes: " + ", ".join(DROP), mujoco=mujoco.__version__)
        (new / f"{NAME}.json").write_text(json.dumps(meta, indent=1) + "\n")
        files = [f"{NAME}.xml", f"{NAME}.json"]
        if check:
            diff = [f for f in files if not (OUT / f).exists() or (new / f).read_bytes() != (OUT / f).read_bytes()]
            print(f"{len(files)} files regenerated; different from the committed ones: {diff or 'none'}")
            sys.exit(1 if diff else 0)
        OUT.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copyfile(new / f, OUT / f)
        print(f"wrote {', '.join(files)} to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
