"""Forest terrain + fixtures check: a terrain build of a model with a fixture (a per-world placed world geom,
forest_engine.step.fixture_groups) against the same model without it, on the forest engine (the tree engine has no
fixtures).

The model: an exported robot (MODEL.xml / MODEL.json, scripts/robots/export_robot.py; default the Apollo of
fixtures/robots) with a box "crate" on the world body, explicit pairs crate - each foot geom with the foot-floor pair's
parameters, and a nablaengine_fixtures text that makes the crate a fixture. The states and the terrain: a terrain_check
reference (REF.npz, the terrain from REF.mjb). Two comparisons, forward (qvel, contact counts) and VJP (qpos, qvel,
ctrl), per world:

  slot   the crate placed per world under the left foot (5 mm into it) by its fixture placement (com's fixture words),
         against the model without the crate whose terrain object slot 0 holds the same box (com's object words): the
         same contact geometry by two code paths (the forest's box-box / capsule-box pairs vs the terrain's object
         contacts), so agreement within float32 rounding of the two CCD ports where the contacts are equal
  offset the crate's fixture placement zero (the crate 5 m under the floor: no contacts) and object slot 0 under the
         foot, against the model without the crate with the same object slot: the object words after the fixture
         words (com_layout); expected bitwise equal outputs

  python scripts/forest_terrain_fixture_check.py --ref runs/terrain/robots/apollo_ref.npz --out runs/forest_gaps/fix.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

HALF = (0.15, 0.12, 0.05)
DEPTH = 5e-3


def crate_model(xml, meta):
    """(model with the crate fixture, model without it)."""
    import mujoco
    base = mujoco.MjModel.from_xml_path(str(xml))
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.worldbody.add_geom(name="crate", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(HALF), pos=[0.0, 0.0, -5.0],
                            contype=0, conaffinity=0)
    for p in meta["foot_pairs"]:
        g = int(base.pair_geom2[p]) if base.geom_bodyid[base.pair_geom1[p]] == 0 else int(base.pair_geom1[p])
        spec.add_pair(name=f"crate_{base.geom(g).name}", geomname1="crate", geomname2=base.geom(g).name,
                      condim=int(base.pair_dim[p]), friction=list(base.pair_friction[p]),
                      solref=list(base.pair_solref[p]), solimp=list(base.pair_solimp[p]),
                      margin=float(base.pair_margin[p]), gap=float(base.pair_gap[p]))
    spec.add_text(name="nablaengine_fixtures", data=json.dumps(
        [dict(name="crate", root="crate", export=[0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0], geoms=["crate"], sites=[],
              roots=[])]))
    return spec.compile(), base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="terrain_check's REF.npz (states; the terrain from REF.mjb)")
    ap.add_argument("--model", default=str(REPO / "fixtures/robots/apollo/apollo_flat"))
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--mode", default="analytic")
    ap.add_argument("--which", choices=("slot", "offset", "both"), default="both")
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dump", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    import mujoco
    from nablaengine import engines, jax_ffi, terrain as T
    meta = json.loads(Path(a.model).with_suffix(".json").read_text())
    mf, m0 = crate_model(Path(a.model).with_suffix(".xml"), meta)
    feet = {id(m0): list(meta["foot_pairs"]),
            id(mf): [mujoco.mj_name2id(mf, mujoco.mjtObj.mjOBJ_PAIR, m0.pair(p).name) for p in meta["foot_pairs"]]}
    assert min(feet[id(mf)]) >= 0, "foot pairs without names"
    r = dict(np.load(a.ref))
    ter = T.Terrain.from_model(mujoco.MjModel.from_binary_path(str(Path(a.ref).with_suffix(".mjb"))))
    B = min(a.nworld, len(r["qpos"]))
    S, it = int(r["substeps"]) if "substeps" in r else 4, int(r["iterations"])
    q0 = np.asarray(r["qpos"][:B], np.float64)
    foot = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, meta["foot_geoms"][0])
    d = mujoco.MjData(m0)
    box = np.zeros((B, T.OBJ_WORDS), np.float32)
    for w in range(B):
        d.qpos[:] = q0[w]
        mujoco.mj_kinematics(m0, d)
        x, R = d.geom_xpos[foot], d.geom_xmat[foot].reshape(3, 3)
        lo = x[2] - np.abs(R[2] * m0.geom_size[foot]).sum() if int(m0.geom_type[foot]) == int(mujoco.mjtGeom.mjGEOM_BOX) \
            else x[2] - m0.geom_size[foot][0] - abs(R[2, 2]) * m0.geom_size[foot][1]
        box[w, :3] = (x[0], x[1], lo + DEPTH - HALF[2])
        box[w, 3:12] = np.eye(3).ravel()
        box[w, 12:15] = HALF
    f = lambda k: jnp.asarray(np.ascontiguousarray(r[k][:B], np.float32))
    fr = f("friction") if "friction" in r else jnp.ones(B, jnp.float32)
    rng = np.random.default_rng(7)
    cq = jnp.asarray(rng.standard_normal((B, m0.nq)).astype(np.float32))
    cv = jnp.asarray(rng.standard_normal((B, m0.nv)).astype(np.float32))

    def run(m, objects, fix_t, slot):
        engines.prefer_forest(m)
        info = jax_ffi.init(model=m, foot_pairs=feet[id(m)], com_body=meta["com_body"],
                            terrain=T.TerrainConfig(geoms=str(r["geoms"]), mode=a.mode, objects=objects))
        jax_ffi.set_terrain(ter)
        lay = info["com_layout"]
        com = np.zeros((B, int(info["com_words"])), np.float32)
        if "com" in r:
            com[:, :3] = r["com"][:B]
        if fix_t is not None:
            o, wd = lay["fixtures"]
            assert wd == 7, lay
            com[:, o:o + 3] = fix_t
            com[:, o + 3] = 1.0
        if slot is not None:
            o, wd = lay["objects"]
            com[:, o:o + T.OBJ_WORDS] = slot
        com = jnp.asarray(com)
        out = jax_ffi.forward(f("qpos"), f("qvel"), f("warm"), f("ctrl"), fr, com, iterations=it, substeps=S,
                              contacts=True)
        ct = jax_ffi.contact_table(out[4])

        def loss(qpos, qvel, ctrl):
            q, v, _ = jax_ffi.step(qpos, qvel, f("warm"), ctrl, fr, com, iterations=it, substeps=S)
            return jnp.sum(q * cq) + jnp.sum(v * cv)
        g = jax.grad(loss, (0, 1, 2))(f("qpos"), f("qvel"), f("ctrl"))
        return dict(layout=json.dumps({k: list(v) for k, v in lay.items()}), qvel=np.asarray(out[1]),
                    qpos=np.asarray(out[0]), ncon=(np.asarray(ct["pair"]) >= 0).sum(1), info=np.asarray(out[3]),
                    g0=np.asarray(g[0]), g1=np.asarray(g[1]), g2=np.asarray(g[2]))

    def rel(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        return np.linalg.norm(x - y, axis=1) / np.maximum(np.linalg.norm(y, axis=1), 1e-12)

    def cmp(A, Bm, name):
        eq = A["ncon"] == Bm["ncon"]
        rv = rel(A["qvel"], Bm["qvel"])
        res = dict(layout_fixture=A["layout"], layout_ref=Bm["layout"], ncon_equal=float(eq.mean()),
                   ncon_mean=[float(A["ncon"].mean()), float(Bm["ncon"].mean())],
                   qvel_rel_median=float(np.median(rv)), qvel_rel_median_ncon_equal=float(np.median(rv[eq])),
                   qvel_rel_p99=float(np.percentile(rv, 99)),
                   qvel_bitwise=bool(np.array_equal(A["qvel"], Bm["qvel"])),
                   qpos_bitwise=bool(np.array_equal(A["qpos"], Bm["qpos"])),
                   grad_rel_median={k: float(np.median(rel(x, y))) for k, x, y in zip(("q", "v", "u"), A["g"], Bm["g"])},
                   grad_bitwise=[bool(np.array_equal(x, y)) for x, y in zip(A["g"], Bm["g"])],
                   finite=bool(all(np.isfinite(x).all() for x in [A["qvel"], *A["g"]])))
        print(name, json.dumps(res))
        return res

    if a.variant:
        t = box[:, :3] - np.array([0.0, 0.0, -5.0], np.float32)
        args = {"fix_slot": (mf, 0, t, None), "ref_slot": (m0, 1, None, box),
                "fix_offset": (mf, 1, np.zeros((B, 3), np.float32), box)}[a.variant]
        np.savez(a.dump, **run(*args))
        return
    import subprocess
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="fixture_terrain_"))

    def variant(name):
        out = tmp / f"{name}.npz"
        subprocess.run([sys.executable, __file__, "--ref", a.ref, "--model", a.model, "--nworld", str(a.nworld),
                        "--mode", a.mode, "--variant", name, "--dump", str(out)], check=True)
        z = dict(np.load(out))
        z["g"] = [z.pop("g0"), z.pop("g1"), z.pop("g2")]
        z["layout"] = json.loads(str(z["layout"]))
        return z

    res = dict(ref=str(a.ref), model=str(a.model), nworld=B, mode=a.mode, engine="forest")
    ref = variant("ref_slot")
    if a.which in ("slot", "both"):
        res["slot"] = cmp(variant("fix_slot"), ref, "slot")
    if a.which in ("offset", "both"):
        res["offset"] = cmp(variant("fix_offset"), ref, "offset")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
