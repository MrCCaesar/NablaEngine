"""The two engines' shared conventions (docs/engines.md), without a GPU: routing, the forest engine's capability check
and the generated text of the conventions' opt-ins.

  routing     ($DK_ENGINE unset) the mjlab G1 and the arm scene (an arm with a gripper and a free box,
              scripts/collision/models/midphase_arm_box.xml) go to the forest engine
              (the default); engines.prefer_tree routes a G1-class model to the tree engine and
              prefer_forest to the forest engine, held weakly (a model freed and a new one at its address is not
              routed by it)
  forest      capability.check_forest: the arm scene has no errors; each modified copy raises exactly its own code
              (fluid H22, the CG solver H24a, RK4 T8, contype contacts H15, gear 2 T5, a condim-6 pair H17; a cylinder
              pair, per-joint limit sets, per-dof frictionloss solref / solimp sets, direct solrefs on a pair and a
              joint limit and noslip none; a mixed solref, refsafe off below 2 h and the contact override T13, the
              elliptic cone with solreffriction T14); a condim-6 pair made by envs.convert from contype contacts
              raises H17, and convert.to_engine(condim6_to=3) clears it (RuntimeWarning, a note); the joint-equality
              gripper scripts/eq/models/u1_gripper.xml has no errors, nor its copy with a connect equality, with a fixed
              tendon (limit, stiffness, damping, armature) or a tendon equality; its copies with a connect inactive at
              build or a weld equality raise H25e, a mixed solref T13 (direct solrefs on the equality, a tendon limit
              and a tendon equality none), with noslip T9, with tendon frictionloss H25t, a
              fixed-tendon actuator without a carrier dof T5; a spatial tendon (forest_engine.spatial) and a motor on
              it none, one over a jointless body that a frame sensor and a force sensor name none through the
              converter (sensors retargeted / dropped, the body fuses; T3 and H25 without the converter), a spatial
              tendon with armature or 1300 tendon motors (the per-world storage) H25, 1101 tendons over the
              strided storage budget but inside the packed one none (forest_engine.spatial.compact), a muscle on it
              none (converted by convert.to_engine; also with MuJoCo's muscle parameters set on the converted motor),
              a muscle and a filter actuator on a hinge none, a muscle under implicitfast and a user dyntype T12
              (forest_engine.muscle); engines.require_forest raises UnsupportedModel (a ValueError)
              naming the code, and $DK_ALLOW_UNSUPPORTED=<code> turns it into a RuntimeWarning; the size limits:
              hinge-chain models with nv 65 / 66 bodies and nv 127 / 128 bodies have no errors and no warnings, nv
              128 / 129 bodies raises T1, nv 129 H4, nv 130 / 131 bodies both, and the nb128 unit scene
              (scripts/collision/models/nb128_snake.xml, nv 126, 112 bodies) has no errors and no warnings; a hinge
              chain of 41 links (40 hinge ancestors) has none, of 42 links T4; two free-root chains touching at
              their tips with a contact support of 64 dofs none, of 65 T11 (forest_engine.kdof64); contact margins (forest_engine.margin,
              margin_convex): a margin (and gap) on the arm scene's plane-box, plane-capsule or box-box pair (a
              convex pair) none, a negative margin T10, a box-box margin with $DK_CCD_FLOAT=1 T10
  switch      the explicit routing switch: $DK_ENGINE=tree routes the G1 to the tree engine, forest / auto / unset to
              the forest engine (the default), another value raises; prefer_tree beats $DK_ENGINE=forest, prefer_forest beats
              $DK_ENGINE=tree, the last of the two calls wins, prefer_tree of a multi-tree model raises; the tree
              engine's defaults on the forest engine (F6): jax_ffi._model gives a G1-class model routed by $DK_ENGINE
              the COM-offset body torso_link, one routed by prefer_forest none, and the forest library's constants
              hold torso_link's body id; $DK_CCD_NT reaches a forest build with CCD pairs (F13: CY_NT of the Apollo's
              box-box feet routed to the forest engine); foot_offset / foot_normal on the forest G1 (F14: the text with
              the contacts' pass and the split words, none without the field, the tree engine refuses it); per-element
              sand on the forest G1's terrain build (F15: the text reads the elements' k, 12 words per element and the
              table's sand word, none without terrain_sand, refused without foot_offset); per-world
              foot_torsional on the forest engine (F4): the text of
              a condim-4 G1 with it reads the world's spin friction, a build without it has no trace; the capacity
              counts (F16, init(capacity_diag=True)): the forest G1's text counts the dropped contacts' rows (add_contact,
              k_ccd) and writes info (B, 6), capacity_diag=False gives the plain build's text, the tree engine refuses it
  text        jax_ffi.library_sources: the G1 engine's default build has no CON_WORDS and 8-word FFI contact buffers,
              contact_forces=True gives 12 words and leaves the other translation units' text before the hooks
              unchanged (the hooks only add); the forest engine's build is the same with and without contact_forces;
              info helpers (con_words, com_layout, adj_stage) as documented; the BAM servo model's text
              (actuator=Bam() on the mjlab G1's hinges) generates on both engines (every hook anchor found); on the forest
              engine a muscle build's text (the activation channel) and the packed tendon storage's text (TS_JW) of
              a build over the strided budget generate; a forest library with a contact support of 40 dofs has
              forest_engine.kdof64's edits, one with 32 none; the arm scene's library has kbimp's direct solref
              form with a direct pair solref (forest_engine.solref), none with its standard ones; the arm scene's
              library has the free box's implicit gyroscopic block (forest_engine.free_gyro; -DDK_FREE_GYRO=0 from
              cuda.forest_hold_defines on sm_90, not on sm_86), the forest G1's none

  midphase    the forest engine's collision mid-phase (forest_engine.midphase, docs/engines.md): unset / auto /
              0 build the text of a model without it (no MIDPHASE), 1 the node level with the contact sort, 2 the group
              level; a model naming groups (text element nablaengine_midphase) gets them under auto; the run table
              covers every pair once in pair order; on 64 random poses of the arm scene (float64, mj_kinematics)
              every pair that passes the per-pair bound test lies in a run whose node spheres overlap (conservative)

  sass        (opt-in: --sections ...,sass) an audit of the generated SASS that needs a module outside this release;
              without it the section reports itself unavailable (not passed)

  PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/engines_check.py [--out engines_check.json]
"""
import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
ARM_XML = HERE / "collision" / "models" / "midphase_arm_box.xml"


def arm():
    """The arm scene: a 7-joint arm with a slide gripper and a free box (explicit pairs, position servos)."""
    return mujoco.MjModel.from_xml_path(str(ARM_XML))


def g1():
    from nablaengine import fixtures
    return fixtures.load_model(fixtures.fixtures_dir(), check=None)


def sec_routing():
    from nablaengine import engines
    old = os.environ.pop(engines.ENV, None)
    try:
        m, _ = g1()
        res = dict(g1=engines.name(m), arm=engines.name(arm()))
        m2, _ = g1()
        engines.prefer_tree(m2)
        res["prefer_tree"] = engines.name(m2)
        res["prefer_tree_other_model"] = engines.name(m)
        m3, _ = g1()
        engines.prefer_forest(m3)
        res["prefer_forest"] = engines.name(m3)
        del m2, m3
        gc.collect()
        res["weak_entries_after_free"] = len(list(engines._PREFER_FOREST)) + len(list(engines._PREFER_TREE))
    finally:
        if old is not None:
            os.environ[engines.ENV] = old
    ok = (res["g1"] == "forest" and res["arm"] == "forest" and res["prefer_tree"] == "g1"
          and res["prefer_tree_other_model"] == "forest" and res["prefer_forest"] == "forest"
          and res["weak_entries_after_free"] == 0)
    return ok, res


def sec_switch():
    import re
    import numpy as np
    from nablaengine import engines, jax_ffi
    from nablaengine.params import as_params
    old = os.environ.pop(engines.ENV, None)
    res = {}
    try:
        m, mt = g1()
        for v in ("", "auto", "tree", "g1", "forest", "Forest"):
            os.environ[engines.ENV] = v
            res[f"env_{v or 'unset'}"] = engines.name(m)
        os.environ[engines.ENV] = "bogus"
        try:
            engines.name(m)
            res["env_bogus"] = "accepted"
        except ValueError:
            res["env_bogus"] = "ValueError"
        os.environ[engines.ENV] = "forest"
        m1, _ = g1()
        engines.prefer_tree(m1)
        res["prefer_tree_vs_env_forest"] = engines.name(m1)
        engines.prefer_forest(m1)
        res["then_prefer_forest"] = engines.name(m1)
        os.environ[engines.ENV] = "tree"
        res["prefer_forest_vs_env_tree"] = engines.name(m1)
        try:
            engines.prefer_tree(arm())
            res["prefer_tree_arm"] = "accepted"
        except ValueError:
            res["prefer_tree_arm"] = "ValueError"
        os.environ[engines.ENV] = "forest"
        m2, _ = g1()
        res["f6_env_forest_com_body"] = jax_ffi._model(model=m2, foot_pairs=mt["foot_pairs"])[2]
        res["f6_fixtures_com_body"] = jax_ffi._model()[2]
        os.environ.pop(engines.ENV, None)
        m5, _ = g1()
        res["f6_default_engine"] = engines.name(m5)
        res["f6_default_com_body"] = jax_ffi._model(model=m5, foot_pairs=mt["foot_pairs"])[2]
        os.environ[engines.ENV] = "forest"
        m3, _ = g1()
        engines.prefer_forest(m3)
        res["f6_prefer_forest_com_body"] = jax_ffi._model(model=m3, foot_pairs=mt["foot_pairs"])[2]
        E = engines.modules(m2)
        c_t = E.source.constants(m2, res["f6_env_forest_com_body"])[0]
        c_n = E.source.constants(m2, None)[0]
        tid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        d = np.flatnonzero(c_t != c_n)
        res["f6_forest_torso_id"] = bool(len(d) == 1 and c_t[d[0]] == tid and c_n[d[0]] == -1)
        root = Path(__file__).resolve().parents[1] / "fixtures" / "robots" / "apollo"
        meta = json.loads((root / "apollo_flat.json").read_text())
        ap_m = mujoco.MjModel.from_xml_path(str(root / "apollo_flat.xml"))
        engines.prefer_forest(ap_m)
        nt_old = os.environ.get("DK_CCD_NT")
        try:
            for nt in ("1", "4", "12"):
                os.environ["DK_CCD_NT"] = nt
                u = jax_ffi.library_sources(ap_m, meta["foot_pairs"], com_body=meta["com_body"])[0]["dk_kv0_n0.cu"]
                res[f"f13_ccd_nt_{nt}"] = re.findall(r"#define CY_NT (\d+)", u)
        finally:
            os.environ.pop("DK_CCD_NT") if nt_old is None else os.environ.__setitem__("DK_CCD_NT", nt_old)
        m4, _ = g1()
        for p in range(m4.npair):
            if m4.pair_dim[p] >= 3:
                m4.pair_dim[p], m4.pair_friction[p, 2] = 4, 0.02
        u_t = jax_ffi.library_sources(m4, mt["foot_pairs"], com_body="torso_link",
                                      params=as_params(("foot_solref", "foot_torsional")))[0]
        u_s = jax_ffi.library_sources(m4, mt["foot_pairs"], com_body="torso_link", params=as_params(("foot_solref",)))[0]
        res["f4_musw_reads"] = sum(v.count("s.musw") for v in u_t.values())
        res["f4_without_field"] = sum(v.count("musw") for v in u_s.values())
        res["f4_sum"] = "else if (q == 7) s.gpw[PW_TORS" in u_t["dk_kv0_n3.cu"]
        mo, _ = g1()
        engines.prefer_forest(mo)
        sp_o = jax_ffi.contact_spec(mo, foot_normal=True)
        u_o = jax_ffi.library_sources(mo, mt["foot_pairs"], com_body="torso_link", contact_forces=sp_o,
                                      params=as_params(("foot_solref", "foot_offset")))[0]
        u_n = jax_ffi.library_sources(mo, mt["foot_pairs"], com_body="torso_link",
                                      params=as_params(("foot_solref",)))[0]
        res["f14_pass"] = sum(v.count("foff_contacts(s, lane);") for v in u_o.values())
        res["f14_split_words"] = "#define CF_FSPLIT 1" in u_o["dk_kv0_n3.cu"] and sp_o.con_words == 14
        res["f14_without_field"] = sum(v.count("foff_") + v.count("CF_FSPLIT") for v in u_n.values())
        mt_, _ = g1()
        engines.prefer_tree(mt_)
        try:
            jax_ffi.library_sources(mt_, mt["foot_pairs"], params=as_params(("foot_offset",)))
            res["f14_tree_refuses"] = False
        except ValueError:
            res["f14_tree_refuses"] = True
        from nablaengine import terrain as T
        cs, c0 = T.TerrainConfig(geoms="feet", terrain_sand=True), T.TerrainConfig(geoms="feet")
        sp_s = jax_ffi.contact_spec(mo, foot_normal=True, terrain=cs)
        u_s = jax_ffi.library_sources(mo, mt["foot_pairs"], com_body="torso_link", contact_forces=sp_s, terrain=cs,
                                      params=as_params(("foot_offset",)))[0]
        sp_0 = jax_ffi.contact_spec(mo, foot_normal=True, terrain=c0)
        u_0 = jax_ffi.library_sources(mo, mt["foot_pairs"], com_body="torso_link", contact_forces=sp_0, terrain=c0,
                                      params=as_params(("foot_offset",)))[0]
        res["f15_sand_reads"] = sum(v.count("foff_k(") for v in u_s.values())
        res["f15_words"] = ("#define TER_MW 12" in u_s["dk_kv0_n3.cu"] and "#define CF_FSAND 1" in u_s["dk_kv0_n3.cu"]
                            and sp_s.con_words == sp_0.con_words + 1)
        res["f15_without"] = sum(v.count("foff_k") + v.count("CF_FSAND") + v.count("TER_MW 12") for v in u_0.values())
        try:
            jax_ffi.library_sources(mo, mt["foot_pairs"], com_body="torso_link", terrain=cs,
                                    params=as_params(("foot_solref",)))
            res["f15_needs_foot_offset"] = False
        except ValueError:
            res["f15_needs_foot_offset"] = True
        u_c = jax_ffi.library_sources(mo, mt["foot_pairs"], com_body="torso_link", capacity_diag=True)[0]
        u_p = jax_ffi.library_sources(mo, mt["foot_pairs"], com_body="torso_link")[0]
        u_f = jax_ffi.library_sources(mo, mt["foot_pairs"], com_body="torso_link", capacity_diag=False)[0]
        res["f16_counts"] = [sum(v.count("atomicAdd(&s.cap_xrows") for v in u_c.values()),
                             u_c["dk_kv0_n3.cu"].count("atomicAdd(&g.cap_xrows"),
                             u_c["dk_kv0_n3.cu"].count("info[w * 6 + 5]"), u_c["dk_ffi.cu"].count("dims_are(*info, {B, 6})")]
        res["f16_without"] = sum(v.count("cap_xrows") + v.count("cap_con") + v.count("{B, 6}") for v in u_p.values())
        res["f16_off_is_plain"] = u_f == u_p
        try:
            jax_ffi.library_sources(mt_, mt["foot_pairs"], capacity_diag=True)
            res["f16_tree_refuses"] = False
        except ValueError:
            res["f16_tree_refuses"] = True
    finally:
        os.environ.pop(engines.ENV, None)
        if old is not None:
            os.environ[engines.ENV] = old
    ok =(res["env_tree"] == res["env_g1"] == "g1"
          and res["env_unset"] == res["env_auto"] == res["env_forest"] == res["env_Forest"] == "forest" and res["env_bogus"] == "ValueError"
          and res["prefer_tree_vs_env_forest"] == "g1" and res["then_prefer_forest"] == "forest"
          and res["prefer_forest_vs_env_tree"] == "forest" and res["prefer_tree_arm"] == "ValueError"
          and res["f6_env_forest_com_body"] == "torso_link" and res["f6_fixtures_com_body"] == "torso_link"
          and res["f6_default_engine"] == "forest" and res["f6_default_com_body"] == "torso_link"
          and res["f6_prefer_forest_com_body"] is None and res["f6_forest_torso_id"]
          and all(res[f"f13_ccd_nt_{nt}"] == [nt] for nt in ("1", "4", "12"))
          and res["f4_musw_reads"] > 0 and res["f4_without_field"] == 0 and res["f4_sum"]
          and res["f14_pass"] > 0 and res["f14_split_words"] and res["f14_without_field"] == 0
          and res["f14_tree_refuses"]
          and res["f15_sand_reads"] > 0 and res["f15_words"] and res["f15_without"] == 0
          and res["f15_needs_foot_offset"]
          and min(res["f16_counts"]) > 0 and res["f16_without"] == 0 and res["f16_off_is_plain"]
          and res["f16_tree_refuses"])
    return bool(ok), res


def _sized(nh, nfree, chain=16):
    body, k = [], 0
    while k < nh:
        n = min(chain, nh - k)
        body += [f'<body pos="{0.3 * (k // 16)} 0 1"><joint type="hinge" axis="0 1 0"/>'
                 f'<geom type="sphere" size="0.02" contype="0" conaffinity="0"/>'] + \
                ['<body pos="0.05 0 0"><joint type="hinge" axis="0 1 0"/>'
                 '<geom type="sphere" size="0.02" contype="0" conaffinity="0"/>'] * (n - 1) + ["</body>"] * n
        k += n
    body += [f'<body pos="{0.1 * i} 1 1"><freejoint/><geom type="sphere" size="0.02" contype="0" conaffinity="0"/>'
             f'</body>' for i in range(nfree)]
    return f'<mujoco><option integrator="implicitfast"/><worldbody>{"".join(body)}</worldbody></mujoco>'


def _tips(n1, n2):
    def chain(n, y):
        return (f'<body pos="0 {y} 1"><freejoint/><geom type="capsule" size="0.02" fromto="0 0 0 0.05 0 0"/>'
                + '<body pos="0.05 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" size="0.02" '
                  'fromto="0 0 0 0.05 0 0"/>' * n + f'<geom name="tip{y}" type="sphere" size="0.02"/>'
                + "</body>" * (n + 1))
    return (f'<mujoco><option integrator="implicitfast"/><default><geom contype="0" conaffinity="0"/></default>'
            f'<worldbody>{chain(n1, 0)}{chain(n2, 1)}</worldbody><contact><pair geom1="tip0" geom2="tip1"/></contact>'
            f'</mujoco>')


def sec_forest():
    from nablaengine import engines
    from nablaengine.model import capability
    I, Sv = mujoco.mjtIntegrator, mujoco.mjtSolver

    def ctype(m):
        m.geom_contype[:] = 1
        m.geom_conaffinity[:] = 1

    def gear(m):
        m.actuator_gear[0, 0] = 2.0

    def cyl(m):
        g = int(m.pair_geom2[0]) if m.geom_bodyid[m.pair_geom2[0]] else int(m.pair_geom1[0])
        m.geom_type[g] = int(mujoco.mjtGeom.mjGEOM_CYLINDER)

    def lim_joint(m):
        return [j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE][-1]

    def limit_set(m):
        m.jnt_solref[lim_joint(m)] = (0.005, 1.0)

    def pair_of(m, types):
        G = mujoco.mjtGeom
        want = tuple(sorted(int(getattr(G, f"mjGEOM_{t}")) for t in types))
        return next(p for p in range(m.npair)
                    if tuple(sorted((int(m.geom_type[m.pair_geom1[p]]), int(m.geom_type[m.pair_geom2[p]])))) == want)

    def margin(types, mg=0.001, gap=0.0):
        def f(m):
            p = pair_of(m, types)
            m.pair_margin[p], m.pair_gap[p] = mg, gap
        return f

    def floss_sets(m):
        m.dof_frictionloss[:2] = 0.5
        m.dof_solimp[1, :2] = (0.99, 0.999)
        m.dof_solref[1, 0] = 0.01

    cases = {"fluid": (lambda m: setattr(m.opt, "density", 1.2), ["H22"]),
             "cg": (lambda m: setattr(m.opt, "solver", int(Sv.mjSOL_CG)), ["H24a"]),
             "noslip": (lambda m: setattr(m.opt, "noslip_iterations", 3), []),
             "rk4": (lambda m: setattr(m.opt, "integrator", int(I.mjINT_RK4)), ["T8"]),
             "contype": (ctype, ["H15"]), "gear": (gear, ["T5"]), "cylinder": (cyl, []),
             "limit_sets": (limit_set, []),
             "floss_sets": (floss_sets, []),
             "limit_margin": (lambda m: m.jnt_margin.__setitem__(lim_joint(m), 0.01), ["T0"]),
             "condim6": (lambda m: m.pair_dim.__setitem__(0, 6), ["H17"]),
             "margin_plane_box": (margin(("PLANE", "BOX"), 0.001, 0.002), []),
             "margin_plane_capsule": (margin(("PLANE", "CAPSULE")), []),
             "margin_box_box": (margin(("BOX", "BOX")), []),
             "gap_box_box": (margin(("BOX", "BOX"), 0.0, 0.001), []),
             "margin_negative": (margin(("PLANE", "BOX"), -0.001), ["T10"]),
             "solref_direct": (lambda m: (m.pair_solref.__setitem__(0, (-5000.0, -50.0)),
                                          m.jnt_solref.__setitem__(lim_joint(m), (-2000.0, -30.0))), []),
             "solref_mixed": (lambda m: m.pair_solref.__setitem__(0, (0.02, 0.0)), ["T13"]),
             "solref_refsafe": (lambda m: (m.pair_solref.__setitem__(0, (0.001, 1.0)),
                                           setattr(m.opt, "disableflags", int(m.opt.disableflags)
                                                   | int(mujoco.mjtDisableBit.mjDSBL_REFSAFE))), ["T13"]),
             "solref_override": (lambda m: setattr(m.opt, "enableflags", int(m.opt.enableflags)
                                                   | int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)), ["T13"]),
             "elliptic_solreffriction": (lambda m: (setattr(m.opt, "cone", int(mujoco.mjtCone.mjCONE_ELLIPTIC)),
                                                    m.pair_solreffriction.__setitem__(0, (0.01, 1.0))), ["T14"])}
    res, ok = {"base": capability.check_forest(arm()).codes()}, True
    ok &= res["base"] == []
    for name, (f, want) in cases.items():
        m = arm()
        f(m)
        got = capability.check_forest(m).codes()
        res[name] = got
        ok &= got == want
    old_ccdf = os.environ.get("DK_CCD_FLOAT")
    os.environ["DK_CCD_FLOAT"] = "1"
    try:
        m = arm()
        margin(("BOX", "BOX"))(m)
        res["margin_box_box_ccd_float"] = got = capability.check_forest(m).codes()
        ok &= got == ["T10"]
    finally:
        if old_ccdf is None:
            os.environ.pop("DK_CCD_FLOAT")
        else:
            os.environ["DK_CCD_FLOAT"] = old_ccdf
    from nablaengine.envs import convert
    u1 = (Path(__file__).resolve().parent / "eq" / "models" / "u1_gripper.xml").read_text()
    eq_cases = {"eq_joint": (u1, []),
                "eq_connect": (u1.replace("</equality>", '<connect body1="left" body2="right" anchor="0 0 0"/></equality>'),
                               []),
                "eq_connect_inactive": (u1.replace("</equality>", '<connect body1="left" body2="right" anchor="0 0 0" '
                                                   'active="false"/></equality>'), ["H25e"]),
                "eq_noslip": (u1.replace("<option ", '<option noslip_iterations="3" '), ["T9"]),
                "eq_weld": (u1.replace("</equality>", '<weld body1="left" body2="right"/></equality>'), ["H25e"]),
                "eq_solref_direct": (u1.replace('solref="0.005 1"', 'solref="-1000 -10"'), []),
                "eq_solref_mixed": (u1.replace('solref="0.005 1"', 'solref="0.005 -10"'), ["T13"]),
                "tendon_fixed": (u1.replace("<actuator>", '<tendon><fixed name="t" limited="true" range="0 0.04" '
                                            'stiffness="1" damping="0.1" armature="0.01"><joint joint="left" coef="0.5"/>'
                                            '<joint joint="right" coef="0.5"/></fixed></tendon><actuator>'), []),
                "tendon_eq": (u1.replace("<actuator>", '<tendon><fixed name="t"><joint joint="left" coef="1"/></fixed>'
                                         '</tendon><actuator>').replace("</equality>", '<tendon tendon1="t"/></equality>'),
                              []),
                "tendon_solref_direct": (u1.replace("<actuator>", '<tendon><fixed name="t" limited="true" range="0 0.04" '
                                                    'solreflimit="-2000 -20"><joint joint="left" coef="0.5"/><joint '
                                                    'joint="right" coef="0.5"/></fixed></tendon><actuator>')
                                         .replace("</equality>", '<tendon tendon1="t" solref="-3000 -40"/></equality>'),
                                         []),
                "tendon_solref_mixed": (u1.replace("<actuator>", '<tendon><fixed name="t" limited="true" range="0 0.04" '
                                                   'solreflimit="-2000 0.5"><joint joint="left" coef="0.5"/><joint '
                                                   'joint="right" coef="0.5"/></fixed></tendon><actuator>'), ["T13"]),
                "tendon_frictionloss": (u1.replace("<actuator>", '<tendon><fixed name="t" frictionloss="0.1"><joint '
                                                   'joint="left" coef="1"/></fixed></tendon><actuator>'), ["H25t"]),
                "tendon_spatial": (u1.replace('<geom name="left_pad"', '<site name="sl"/><geom name="left_pad"')
                                   .replace('<geom name="right_pad"', '<site name="sr"/><geom name="right_pad"')
                                   .replace("<actuator>", '<tendon><spatial name="t"><site site="sl"/><site site="sr"/>'
                                            '</spatial></tendon><actuator>'), []),
                "tendon_spatial_motor": (u1.replace('<geom name="left_pad"', '<site name="sl"/><geom name="left_pad"')
                                         .replace('<geom name="right_pad"', '<site name="sr"/><geom name="right_pad"')
                                         .replace("<actuator>", '<tendon><spatial name="t" limited="true" '
                                                  'range="0.001 0.2"><site site="sl"/><site site="sr"/></spatial>'
                                                  '</tendon><actuator><motor tendon="t" gear="2"/>'), []),
                "tendon_spatial_armature": (u1.replace('<geom name="left_pad"', '<site name="sl"/><geom name="left_pad"')
                                            .replace('<geom name="right_pad"', '<site name="sr"/><geom name="right_pad"')
                                            .replace("<actuator>", '<tendon><spatial name="t" armature="0.01"><site '
                                                     'site="sl"/><site site="sr"/></spatial></tendon><actuator>'),
                                            ["H25"]),
                "tendon_actuator": (u1.replace("<actuator>", '<tendon><fixed name="t"><joint joint="left" coef="1"/>'
                                               '</fixed></tendon><actuator><motor tendon="t"/>'), ["T5"])}
    for name, (xml, want) in eq_cases.items():
        got = capability.check_forest(convert.to_engine(xml, {}).model).codes()
        res[name] = got
        ok &= got == want
    sb = u1.replace('<geom name="left_pad" type="box" size="0.02 0.005 0.02" mass="0.1"/>',
                    '<geom name="left_pad" type="box" size="0.02 0.005 0.02" mass="0.1"/><body name="plate" '
                    'pos="0.01 0 0.02"><geom name="plate" type="box" size="0.01 0.01 0.002" pos="0.003 0 0" '
                    'mass="0.02" contype="0" conaffinity="0"/><site name="sp"/></body>').replace(
        '<geom name="right_pad"', '<site name="sr"/><geom name="right_pad"').replace(
        "<actuator>", '<tendon><spatial name="t"><site site="sp"/><site site="sr"/></spatial></tendon><actuator>'
        '<motor tendon="t"/>').replace("</mujoco>", '<sensor><framelinvel name="pv" objtype="body" objname="plate"/>'
                                       '<force name="pf" site="sp"/></sensor></mujoco>')
    cv = convert.to_engine(sb, {})
    raw = mujoco.MjSpec.from_string(sb)
    raw.compiler.fusestatic = True
    res["tendon_sensor_static_body"] = dict(
        codes=capability.check_forest(cv.model).codes(), nsensor=int(cv.model.nsensor),
        notes=sum(n.startswith("sensors on jointless bodies") for n in cv.notes),
        source=capability.check_forest(raw.compile()).codes())
    ok &= res["tendon_sensor_static_body"] == dict(codes=[], nsensor=1, notes=2, source=["H15", "H25", "T3"])
    many = eq_cases["tendon_spatial"][0].replace(
        '<tendon><spatial name="t"><site site="sl"/><site site="sr"/></spatial></tendon><actuator>',
        "<tendon>" + "".join(f'<spatial name="t{k}"><site site="sl"/><site site="sr"/></spatial>' for k in range(1300))
        + "</tendon><actuator>" + "".join(f'<motor tendon="t{k}"/>' for k in range(1300)))
    res["tendon_storage"] = capability.check_forest(convert.to_engine(many, {}).model).codes()
    ok &= res["tendon_storage"] == ["H25"]
    from nablaengine.forest_engine import spatial
    pk = convert.to_engine(_packed_xml(u1), {}).model
    res["tendon_storage_packed"] = dict(codes=capability.check_forest(pk).codes(), packed=spatial.compact_on(pk),
                                        words=spatial.words(pk))
    ok &= res["tendon_storage_packed"] == dict(codes=[], packed=True, words=2209)
    eul = lambda x: x.replace('integrator="implicitfast"', 'integrator="Euler"')
    mus = eq_cases["tendon_spatial_motor"][0].replace('<motor tendon="t" gear="2"/>', '<muscle tendon="t"/>')
    try:
        res["muscle_convert"] = capability.check_forest(convert.to_engine(eul(mus), {}).model).codes()
    except ValueError as e:
        res["muscle_convert"] = str(e)[:200]
    ok &= res["muscle_convert"] == []
    gen_cases = {"muscle_joint": (eul(u1).replace("<actuator>", '<actuator><muscle joint="left" '
                                                  'lengthrange="0 0.05"/>'), []),
                 "filter_joint": (u1.replace("<actuator>", '<actuator><general joint="left" dyntype="filter" '
                                             'dynprm="0.02" gainprm="5"/>'), []),
                 "muscle_implicitfast": (mus, ["T12"]),
                 "dyn_user": (u1.replace("<actuator>", '<actuator><general joint="left" dyntype="user" dynprm="1" '
                                         'gainprm="5"/>'), ["T12"])}
    for name, (xml, want) in gen_cases.items():
        try:
            m_ = convert.to_engine(xml, {}).model
            got = capability.check_forest(m_).codes()
        except (ValueError, AssertionError) as e:
            got = f"{type(e).__name__}: {str(e)[:200]}"
        res[name] = got
        ok &= got == want
    sp = mujoco.MjSpec.from_string(convert.to_engine(eul(eq_cases["tendon_spatial_motor"][0]), {}).xml)
    for a in sp.actuators:
        if a.trntype != mujoco.mjtTrn.mjTRN_TENDON:
            continue
        a.dyntype, a.gaintype, a.biastype = (mujoco.mjtDyn.mjDYN_MUSCLE, mujoco.mjtGain.mjGAIN_MUSCLE,
                                             mujoco.mjtBias.mjBIAS_MUSCLE)
        a.gainprm = [0.75, 1.05, -1, 200, 0.5, 1.6, 1.5, 1.3, 1.2, 0]
        a.biasprm = [0.75, 1.05, -1, 200, 0.5, 1.6, 1.5, 1.3, 1.2, 0]
        a.dynprm = [0.01, 0.04, 0, 0, 0, 0, 0, 0, 0, 0]
        a.lengthrange = [0.01, 0.2]
        a.gear = [1, 0, 0, 0, 0, 0]
    res["muscle_codes"] = capability.check_forest(sp.compile()).codes()
    ok &= res["muscle_codes"] == []
    c6 = u1.replace('<geom name="left_pad"', '<geom condim="6" name="left_pad"')
    cv = convert.to_engine(c6, {})
    res["condim6_converted"] = dict(codes=capability.check_forest(cv.model).codes(),
                                    condim6_pairs=int((cv.model.pair_dim == 6).sum()))
    ok &= res["condim6_converted"]["codes"] == ["H17"] and res["condim6_converted"]["condim6_pairs"] >= 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cv = convert.to_engine(c6, {}, condim6_to=3)
    res["condim6_to_3"] = dict(codes=capability.check_forest(cv.model).codes(),
                               condim=sorted({int(x) for x in cv.model.pair_dim}),
                               warned=any(x.category is RuntimeWarning and "condim6_to" in str(x.message) for x in w),
                               noted=any("condim 6 -> 3" in n for n in cv.notes))
    ok &= res["condim6_to_3"] == dict(codes=[], condim=[3], warned=True, noted=True)
    size_cases = {"nv65_nb66": ((65, 0), []), "nv127_nb128": ((127, 0), []), "nv128_nb129": ((128, 0), ["T1"]),
                  "nv129_nb25": ((3, 21), ["H4"]), "nv130_nb131": ((130, 0), ["H4", "T1"]),
                  "anc40": ((41, 0, 41), []), "anc41": ((42, 0, 42), ["T4"])}
    for name, (sz, want) in size_cases.items():
        m = mujoco.MjModel.from_xml_string(_sized(*sz))
        nh, nfree = sz[:2]
        rep = capability.check_forest(m)
        res[name] = dict(nv=int(m.nv), nbody=int(m.nbody), errors=rep.codes(), warnings=rep.codes("warn"))
        ok &= rep.codes() == want and not rep.codes("warn")
    m = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parent / "collision" / "models" / "nb128_snake.xml"))
    rep = capability.check_forest(m)
    res["nb128_snake"] = dict(nv=int(m.nv), nbody=int(m.nbody), errors=rep.codes(), warnings=rep.codes("warn"))
    ok &= (m.nv, m.nbody) == (126, 112) and rep.codes() == [] and not rep.codes("warn")
    from nablaengine.forest_engine import step as _fst
    for name, (n1, n2, want) in {"support64": (26, 26, []), "support65": (27, 26, ["T11"])}.items():
        m = mujoco.MjModel.from_xml_string(_tips(n1, n2))
        res[name] = dict(support=int(_fst.max_support(m)), errors=capability.check_forest(m).codes())
        ok &= res[name] == dict(support=12 + n1 + n2, errors=want)
    m = arm()
    gear(m)
    try:
        engines.require_forest(m, None, "engines_check")
        res["require"] = "accepted"
        ok = False
    except capability.UnsupportedModel as e:
        res["require"] = f"UnsupportedModel {e.report.codes()} ValueError={isinstance(e, ValueError)}"
        ok &= isinstance(e, ValueError) and e.report.codes() == ["T5"]
    os.environ[capability.ALLOW_VAR] = "T5"
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engines.require_forest(m, None, "engines_check")
        res["allow"] = [str(x.category.__name__) for x in w]
        ok &= any(x.category is RuntimeWarning for x in w)
    finally:
        del os.environ[capability.ALLOW_VAR]
    return bool(ok), res


def _packed_xml(u1):
    x = (u1.replace('<geom name="left_pad"', '<site name="sl"/><geom name="left_pad"')
         .replace('<geom name="box"', '<site name="sb"/><geom name="box"')
         .replace('<geom name="floor"', '<site name="sw" pos="0 0.2 0.1"/><geom name="floor"'))
    return x.replace("<actuator>", "<tendon>" + '<spatial name="tb"><site site="sw"/><site site="sb"/></spatial>'
                     + "".join(f'<spatial name="k{k}" stiffness="0.01"><site site="sw"/><site site="sl"/></spatial>'
                               for k in range(1100)) + '</tendon><actuator><motor tendon="tb"/>')


def sec_text():
    from nablaengine import engines, jax_ffi
    m, mt = g1()
    engines.prefer_tree(m)
    fp = mt["foot_pairs"]
    u0, c0, d0 = jax_ffi.library_sources(m, fp)
    u1, c1, d1 = jax_ffi.library_sources(m, fp, contact_forces=True)
    res = dict(default_con_words=jax_ffi.con_words(m), cf_con_words=jax_ffi.con_words(m, True),
               default_has_con_words="CON_WORDS" in "".join(u0.values()),
               cf_ffi_12="MAXCON * 12" in u1["dk_ffi.cu"] and "MAXCON * 8" not in u1["dk_ffi.cu"],
               default_ffi_8="MAXCON * 8" in u0["dk_ffi.cu"], same_consts=bool((c0 == c1).all()), same_defines=d0 == d1,
               same_units=sorted(u0) == sorted(u1), com_layout=jax_ffi.com_layout(m))
    from nablaengine import terrain as _ter
    sp = jax_ffi.contact_spec(m, "substeps", True, True)
    u2, c2, _ = jax_ffi.library_sources(m, fp, contact_forces=sp)
    u3, _, _ = jax_ffi.library_sources(m, fp, terrain=_ter.TerrainConfig(), contact_forces=True)
    res.update(spec_con_words=sp.con_words, spec_feet=list(sp.feet), spec_same_consts=bool((c0 == c2).all()),
               spec_ffi_width="(int64_t)substeps * MAXCON * 15" in u2["dk_ffi.cu"],
               terrain_cf_elem="#define CF_TERRAIN 1" in "".join(u3.values()))
    f = arm()
    uf0, _, _ = jax_ffi.library_sources(f, [])
    uf1, _, _ = jax_ffi.library_sources(f, [], contact_forces=True)
    res["forest_same_with_contact_forces"] = uf0 == uf1
    res["forest_con_words"] = jax_ffi.con_words(f)
    from nablaengine import engines
    st = os.environ.pop("DK_ADJ_STAGE", None)
    try:
        res["adj_stage_default"] = [jax_ffi.adj_stage(engines.modules(x)) for x in (m, f)]
    finally:
        if st is not None:
            os.environ["DK_ADJ_STAGE"] = st
    from nablaengine import actuators
    from nablaengine.actuators import Bam
    res["bam_text"] = {}
    for eng in ("forest", "tree"):
        md, mdt = g1()
        spec = actuators.resolve(Bam(), None, "engines_check")
        md = actuators.prepare_model(spec, md)
        (engines.prefer_forest if eng == "forest" else engines.prefer_tree)(md)
        try:
            ub, _, _ = jax_ffi.library_sources(md, mdt["foot_pairs"], com_body="torso_link", bam=spec.bam)
            res["bam_text"][eng] = "#define DK_BAM 1" in "".join(ub.values()) and "DkStepAct" in ub["dk_ffi.cu"]
        except Exception as e:
            res["bam_text"][eng] = f"{type(e).__name__}: {str(e)[:160]}"
    from nablaengine.envs import convert
    u1x = (Path(__file__).resolve().parent / "eq" / "models" / "u1_gripper.xml").read_text()
    sp1 = (u1x.replace('integrator="implicitfast"', 'integrator="Euler"')
           .replace('<geom name="left_pad"', '<site name="sl"/><geom name="left_pad"')
           .replace('<geom name="right_pad"', '<site name="sr"/><geom name="right_pad"')
           .replace("<actuator>", '<tendon><spatial name="t"><site site="sl"/><site site="sr"/></spatial></tendon>'
                    '<actuator><muscle tendon="t"/>'))
    res["spatial_text"] = {}
    for name, xml in (("muscle", sp1), ("packed", _packed_xml(u1x))):
        try:
            ms = convert.to_engine(xml, {}).model
            engines.prefer_forest(ms)
            us, _, _ = jax_ffi.library_sources(ms, [])
            txt = "".join(us.values())
            res["spatial_text"][name] = dict(act="DkStepAct" in us["dk_ffi.cu"], packed="#define TS_JW " in txt)
        except Exception as e:
            res["spatial_text"][name] = f"{type(e).__name__}: {str(e)[:160]}"
    res["kdof64_text"] = {}
    for sup, (n1, n2) in ((32, (10, 10)), (40, (14, 14))):
        mk = mujoco.MjModel.from_xml_string(_tips(n1, n2))
        uk, _, _ = jax_ffi.library_sources(mk, [])
        tk = "".join(uk.values())
        res["kdof64_text"][sup] = dict(edits="(kdof64: KDOF > 32)" in tk, kdof=f"#define KDOF {sup}\n" in tk)
    ok_k = res["kdof64_text"] == {32: dict(edits=False, kdof=True), 40: dict(edits=True, kdof=True)}
    res["solref_text"] = {}
    for nm in ("standard", "direct"):
        mf = arm()
        if nm == "direct":
            mf.pair_solref[0] = (-5000.0, -50.0)
        uf, _, _ = jax_ffi.library_sources(mf, [])
        res["solref_text"][nm] = "(forest_engine.solref)" in "".join(uf.values())
    ok_k &= res["solref_text"] == {"standard": False, "direct": True}
    from nablaengine import cuda as _cuda
    uf, _, _ = jax_ffi.library_sources(arm(), [])
    tf = "".join(uf.values())
    mg, mgt = g1()
    engines.prefer_forest(mg)
    ug, _, _ = jax_ffi.library_sources(mg, mgt["foot_pairs"])
    res["free_gyro_text"] = dict(arm="(forest_engine.free_gyro" in tf,
                                 g1="(forest_engine.free_gyro" in "".join(ug.values()),
                                 hold_sm90="-DDK_FREE_GYRO=0" in _cuda.forest_hold_defines("sm_90", tf),
                                 hold_sm86="-DDK_FREE_GYRO=0" in _cuda.forest_hold_defines("sm_86", tf))
    ok_k &= res["free_gyro_text"] == dict(arm=True, g1=False, hold_sm90=True, hold_sm86=False)
    ok = (ok_k and res["bam_text"] == {"forest": True, "tree": True}
          and res["spatial_text"] == {"muscle": dict(act=True, packed=False), "packed": dict(act=False, packed=True)}
          and res["default_con_words"] == 8 and res["cf_con_words"] == 12 and not res["default_has_con_words"]
          and res["cf_ffi_12"] and res["default_ffi_8"] and res["same_consts"] and res["same_defines"]
          and res["same_units"] and res["com_layout"] == {"com_offset": (0, 3)}
          and res["forest_same_with_contact_forces"] and res["forest_con_words"] == 12
          and res["adj_stage_default"] == [3, 3] and res["spec_con_words"] == 15 and res["spec_same_consts"]
          and res["spec_ffi_width"] and res["terrain_cf_elem"] and len(res["spec_feet"]) == 2)
    return ok, res


def _mp_conservative(f, nposes, spread):
    import numpy as np
    from nablaengine.forest_engine import midphase as MP, step
    t = MP.tables(f)
    rs = t["rstart"]
    cg = step.collision_geoms(f)
    d = mujoco.MjData(f)
    rng = np.random.default_rng(0)
    bad, kept = 0, 0
    order = [step.pair_order(f, int(f.pair_geom1[p]), int(f.pair_geom2[p])) for p in range(f.npair)]
    for _ in range(nposes):
        d.qpos[:] = f.qpos0
        for j in range(f.njnt):
            a = f.jnt_qposadr[j]
            if f.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                lo, hi = f.jnt_range[j] if f.jnt_limited[j] else (-3.0, 3.0)
                d.qpos[a] = rng.uniform(lo, hi)
            elif f.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                d.qpos[a:a + 3] += rng.normal(0, spread, 3)
                q = rng.normal(size=4)
                d.qpos[a + 3:a + 7] = q / np.linalg.norm(q)
        mujoco.mj_kinematics(f, d)
        gx, gR = d.geom_xpos[cg], d.geom_xmat[cg].reshape(-1, 3, 3)
        cen = np.asarray([gx[w & 255] + gR[w & 255] @ np.asarray(o[:3]) for w, o in zip(t["nodes"], t["off"])])
        rad = np.asarray([o[3] for o in t["off"]])
        for r, e in enumerate(t["runs"]):
            n1, n2 = e & 255, (e >> 8) & 255
            if (e >> 16) & 1:
                over = (cen[n2] - gx[e >> 17]) @ gR[e >> 17][:, 2] <= rad[n2]
            else:
                over = np.linalg.norm(cen[n2] - cen[n1]) <= rad[n1] + rad[n2]
            for p in range(rs[r], rs[r + 1]):
                g1, g2 = order[p]
                a, b = cg.index(g1), cg.index(g2)
                if f.geom_type[g1] == mujoco.mjtGeom.mjGEOM_PLANE:
                    hit = (gx[b] - gx[a]) @ gR[a][:, 2] <= f.geom_rbound[g2]
                else:
                    hit = np.linalg.norm(gx[b] - gx[a]) <= f.geom_rbound[g1] + f.geom_rbound[g2]
                kept += bool(hit)
                bad += bool(hit and not over)
    return kept, bad


def sec_midphase():
    from nablaengine.forest_engine import adjoint, midphase as MP
    f = arm()
    old = os.environ.pop("DK_MIDPHASE", None)
    res = {}
    try:
        base = adjoint.cuda_source(f)
        texts = {}
        for v in ("auto", "0", "1", "2"):
            os.environ["DK_MIDPHASE"] = v
            texts[v] = adjoint.cuda_source(f)
        os.environ.pop("DK_MIDPHASE")
        res["off_same"] = texts["auto"] == texts["0"] == base and "MIDPHASE" not in base
        res["on_1"] = ("#define MIDPHASE 1" in texts["1"] and "#define MP_GROUPS 0" in texts["1"]
                       and "#if HAS_DEFER || MIDPHASE" in texts["1"])
        res["on_2"] = "#define MP_GROUPS 1" in texts["2"]
        xml = """<mujoco><worldbody><geom name="floor" type="plane" size="2 2 0.1"/>
          <body name="a" pos="0 0 0.5"><freejoint/><geom name="ga" type="box" size="0.1 0.1 0.1"/>
            <body name="a1" pos="0.2 0 0"><joint type="hinge"/><geom name="ga1" type="capsule" size="0.03 0.1"/></body>
            <body name="a2" pos="-0.2 0 0"><joint type="hinge"/><geom name="ga2" type="capsule" size="0.03 0.1"/></body>
          </body>
          <body name="b" pos="1 0 0.5"><freejoint/><geom name="gb" type="sphere" size="0.1"/></body></worldbody>
          <contact><pair geom1="floor" geom2="ga"/><pair geom1="floor" geom2="ga1"/><pair geom1="floor" geom2="ga2"/>
            <pair geom1="floor" geom2="gb"/><pair geom1="ga1" geom2="gb"/><pair geom1="ga2" geom2="gb"/>
            <pair geom1="ga" geom2="gb"/></contact>
          <custom><text name="nablaengine_midphase" data='[["a", "a1", "a2"]]'/></custom></mujoco>"""
        g = mujoco.MjModel.from_xml_string(xml)
        tg = MP.tables(g)
        res["explicit"] = dict(enabled=MP.enabled(g), group_level=MP.group_level(tg),
                               group=["a", "a1", "a2"] in [n for _, n in MP.describe(g)],
                               text="#define MP_GROUPS 1" in adjoint.cuda_source(g))
        rs = MP.tables(f)["rstart"]
        res["runs_cover"] = (rs[0] == 0 and rs[-1] == f.npair and all(a < b for a, b in zip(rs, rs[1:])))
        kept = bad = 0
        for mm, n, spread in ((f, 64, 0.1), (g, 256, 0.3)):
            k, b_ = _mp_conservative(mm, n, spread)
            kept, bad = kept + k, bad + b_
        res["conservative"] = dict(pairs_passing=int(kept), outside_their_run=int(bad))
    finally:
        os.environ.pop("DK_MIDPHASE", None)
        if old is not None:
            os.environ["DK_MIDPHASE"] = old
    ok = (res["off_same"] and res["on_1"] and res["on_2"] and res["runs_cover"]
          and res["conservative"]["outside_their_run"] == 0 and res["conservative"]["pairs_passing"] > 0
          and all(res["explicit"].values()))
    return ok, res


PTXAS = []


def sec_sass():
    """The SASS audit lives in a module outside this release."""
    sys.path.insert(0, str(HERE / "internal"))
    try:
        import engines_sass
    except ImportError:
        return False, dict(skipped="internal module, not in this release")
    return engines_sass.run(PTXAS)


SECTIONS = {"routing": sec_routing, "switch": sec_switch, "forest": sec_forest, "text": sec_text,
            "midphase": sec_midphase, "sass": sec_sass}
DEFAULT = ("routing", "switch", "forest", "text", "midphase")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sections", default=",".join(DEFAULT), help=f"of {','.join(SECTIONS)}")
    ap.add_argument("--ptxas", action="append", default=[], help="sass: extra ptxas LABEL=PATH (repeatable)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    PTXAS.extend(a.ptxas)
    out, ok = {}, True
    for name in a.sections.split(","):
        good, res = SECTIONS[name]()
        out[name] = dict(ok=good, **res)
        ok &= good
        print(f"{name}: {'PASS' if good else 'FAIL'} {json.dumps(res, default=str)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("engines check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
