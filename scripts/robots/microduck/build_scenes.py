"""Build the comparison scenes from model/scene_walk.xml (the Velocity task's robot).

Every scene gets the settings the Velocity task trains with in mjlab 1.3.0
(make_velocity_env_cfg + microduck FULL_COLLISION):
  timestep 0.005, implicitfast, Newton, pyramidal cone, iterations 10, ls_iterations 20,
  ccd_iterations 50, feet condim 3 / friction 1.0 / priority 1.

  scene_mesh.xml            the sole meshes (convex hulls): the reference
  scene_spheresK.xml        K spheres per foot, contype/conaffinity contacts (drop-in for mjlab
                            if FULL_COLLISION's regex also matches the new names)
  scene_spheresK_pairs.xml  the NablaEngine form: no contype contacts at all, one explicit
                            <pair> per floor-sphere, self-collision meshes switched off
"""
import json
import re
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
SRC = MODEL / "scene_walk.xml"
KS = (4, 6, 8)
FOOT_RE = re.compile(r"^(left|right)_foot(_s\d+)?_collision$")


def mjlab_options(spec):
    o = spec.option
    o.timestep = 0.005
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.impratio = 1.0
    o.iterations = 10
    o.ls_iterations = 20
    o.ccd_iterations = 50


def foot_params(g):
    g.condim = 3
    g.priority = 1
    g.friction = [1.0, 0.005, 0.0001]


def base_spec():
    spec = mujoco.MjSpec.from_file(str(SRC))
    mjlab_options(spec)
    for g in spec.geoms:
        if g.name and FOOT_RE.match(g.name):
            foot_params(g)
    return spec


def add_spheres(spec, fits, K):
    """Replace each sole mesh collision geom by K spheres in the ankle body."""
    for side in ("left", "right"):
        mesh = spec.geom(f"{side}_foot_collision")
        body = mesh.parent
        ref = dict(contype=mesh.contype, conaffinity=mesh.conaffinity, group=mesh.group,
                   solref=list(mesh.solref), solimp=list(mesh.solimp), margin=mesh.margin, gap=mesh.gap)
        f = fits[side]["fits"][str(K)]
        assert fits[side]["body"] == body.name, (fits[side]["body"], body.name)
        spec.delete(mesh)
        for i, (c, r) in enumerate(zip(f["centers"], f["radii"])):
            g = body.add_geom()
            g.name = f"{side}_foot_s{i}_collision"
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [r, 0, 0]
            g.pos = c
            g.mass = 0.0          # the visual/inertial stay as exported; collision geoms carry no mass
            g.density = 0.0
            g.rgba = [1.0, 0.4, 0.1, 0.6]
            for k, v in ref.items():
                setattr(g, k, v)
            foot_params(g)


def contact_params(m):
    """Combined floor-foot parameters MuJoCo uses (read from a live contact)."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND"))
    for z in np.linspace(0.13, 0.0, 400):           # lower the robot until a foot touches
        d.qpos[2] = z
        mujoco.mj_forward(m, d)
        if d.ncon:
            c = d.contact[0]
            return dict(condim=int(c.dim), friction=c.friction.tolist(), solref=c.solref.tolist(),
                        solimp=c.solimp.tolist(), margin=0.0, gap=0.0, includemargin=float(c.includemargin))
    raise RuntimeError("no contact found")


def to_pairs(spec, params):
    """NablaEngine form: explicit floor-foot pairs, every contype contact off."""
    feet = [g.name for g in spec.geoms if g.name and FOOT_RE.match(g.name)]
    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0
    for name in feet:
        p = spec.add_pair()
        p.name = f"floor_{name}"
        p.geomname1 = "floor"
        p.geomname2 = name
        p.condim = params["condim"]
        p.friction = params["friction"]
        p.solref = params["solref"]
        p.solimp = params["solimp"]
        p.margin = params["margin"]
        p.gap = params["gap"]
    return feet


def write(spec, name):
    xml = spec.to_xml()
    path = MODEL / name
    path.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(path))      # must recompile from the written file
    return m


def main():
    fits = json.loads((HERE / "fit_results.json").read_text())
    summary = {}
    m = write(base_spec(), "scene_mesh.xml")
    summary["scene_mesh.xml"] = dict(ngeom=m.ngeom, npair=m.npair, nv=m.nv)
    for K in KS:
        spec = base_spec(); add_spheres(spec, fits, K)
        m = write(spec, f"scene_spheres{K}.xml")
        params = contact_params(m)
        summary[f"scene_spheres{K}.xml"] = dict(ngeom=m.ngeom, npair=m.npair, contact_params=params)
        spec = base_spec(); add_spheres(spec, fits, K)
        feet = to_pairs(spec, params)
        m = write(spec, f"scene_spheres{K}_pairs.xml")
        summary[f"scene_spheres{K}_pairs.xml"] = dict(ngeom=m.ngeom, npair=m.npair, feet=feet)
    (HERE / "scenes.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:3000])


if __name__ == "__main__":
    main()
