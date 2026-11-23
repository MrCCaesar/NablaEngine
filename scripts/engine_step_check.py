"""Check the G1 engine's forward (--substeps substeps of one control step) on corpus states against MuJoCo Warp, MuJoCo C
and the MuJoCo C float64 truth."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from nablaengine import checks, cuda, engines, fixtures, provenance, workdir
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source, step, tape
from nablaengine.g1_engine.host import KERNEL, loader

GATES = dict(supported_share=1.0, mjw_qvel_median=5e-6, mjw_qvel_p99=5e-5, mjw_qpos_median=5e-7, mjw_qpos_p99=5e-6,
             truth_qvel_median=5e-6, truth_qvel_p99=5e-5, truth_qpos_median=5e-7, truth_qpos_p99=5e-6,
             mjw_contacts_match=0.999)
TAPE_GATED = ("T1_outputs_tape_on_vs_off", "T2_state_chain", "T3_contact_table", "T4_L_Q", "T5_L_H",
              "T5b_L_H_engine_geometry", "T8_record_coverage")


HOST = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
static std::vector<float> readf(const char* path, size_t n) {
  std::vector<float> v(n);
  FILE* f = fopen(path, "rb");
  if (!f || fread(v.data(), 4, n, f) != n) { fprintf(stderr, "read %s\n", path); exit(1); }
  fclose(f);
  return v;
}
template <class T> static void writef(const char* path, const T* v, size_t n) { FILE* f = fopen(path, "wb"); fwrite(v, sizeof(T), n, f); fclose(f); }
static float* up(const std::vector<float>& h) { float* d; cudaMalloc(&d, h.size() * 4); cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice); return d; }
int main(int argc, char** argv) {  // dir nworld iterations reps substeps [blocks per SM cap [tape 0 / 1]]
  const char* dir = argv[1];
  int nw = atoi(argv[2]), iters = atoi(argv[3]), reps = atoi(argv[4]), nsub = atoi(argv[5]);
  bool use_tape = argc > 7 && atoi(argv[7]) > 0;  // every launch writes the tape; the last one's goes to tape.bin
  char path[1024];
  snprintf(path, sizeof path, "%s/consts.bin", dir);
  std::vector<float> cf = readf(path, NCONST);
  const float* p = cf.data();
LOAD_CONSTS
#define IN(name, n) snprintf(path, sizeof path, "%s/" #name ".bin", dir); float* d_##name = up(readf(path, (size_t)nw * (n)));
  IN(qpos, NQ) IN(qvel, NV) IN(warm, NV) IN(ctrl, NV - 6) IN(fric, 1) IN(com, 3)
  float *oq, *ov, *ow, *oc; int* oi;
  cudaMalloc(&oc, (size_t)nw * MAXCON * 8 * 4);
  cudaMalloc(&oq, (size_t)nw * NQ * 4); cudaMalloc(&ov, (size_t)nw * NV * 4); cudaMalloc(&ow, (size_t)nw * NV * 4);
  cudaMalloc(&oi, (size_t)nw * 16);
  float* tp = nullptr;
  size_t ntape = use_tape ? (size_t)nw * nsub * REC_WORDS : 0;
  if (use_tape) {  // pre-filled with 0xA5 bytes: a word the kernels never write stays 0xA5A5A5A5 (tape check T8)
    cudaMalloc(&tp, ntape * 4);
    cudaMemset(tp, 0xA5, ntape * 4);
  }
  size_t want = 0;  // a cap of N blocks per SM: request more shared memory than needed so that at most N fit
  if (argc > 6 && atoi(argv[6]) > 0) {
    int dev, per_sm; cudaGetDevice(&dev); cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    want = per_sm / atoi(argv[6]) - 1024;
  }
#if DK_SPLIT  // each kernel its own shared memory (struct SimP / SimN), the same cap
  size_t smem_pre = sizeof(SimP) > want ? sizeof(SimP) : want, smem_post = sizeof(SimN) > want ? sizeof(SimN) : want;
  int nblk = 0, nblk_post = 0;
  cudaFuncSetAttribute(k_pre, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_pre);
  cudaFuncSetAttribute(k_post, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_post);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, k_pre, 32, smem_pre);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk_post, k_post, 32, smem_post);
  void* work;
  cudaMalloc(&work, dk_work_bytes(nw));
  auto launch = [&]() {
    dk_forward_split(0, nw, d_qpos, d_qvel, d_warm, d_ctrl, d_fric, d_com, iters, nsub, oq, ov, ow, oi, oc, tp, work,
                     smem_pre, smem_post);
  };
#else
  size_t smem = sizeof(Sim) > want ? sizeof(Sim) : want;
  cudaFuncSetAttribute(k_substep, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int nblk = 0, nblk_post = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, k_substep, 32, smem);
  auto launch = [&]() {
    k_substep<<<nw, 32, smem>>>(nw, d_qpos, d_qvel, d_warm, d_ctrl, d_fric, d_com, iters, nsub, oq, ov, ow, oi, oc, tp);
  };
#endif
  launch();
  cudaDeviceSynchronize();
#ifdef DK_PROFILE
  {  // cycles per phase summed over worlds and substeps, of the first launch only
    unsigned long long hp[NPROF];
    cudaMemcpyFromSymbol(hp, g_prof, sizeof hp);
    printf("{\"prof_cycles\": [");
    for (int k = 0; k < NPROF; k++) printf("%s%llu", k ? ", " : "", hp[k]);
    printf("]}\n");
  }
#endif
#if DK_SPLIT
  if (getenv("DK_SPLIT_ONLY")) dk_split_only = atoi(getenv("DK_SPLIT_ONLY"));  // timing of one kernel (after the run)
#endif
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < (reps ? 3 : 0); i++)  // warm-up launches (clocks, caches) before timing
    launch();
  cudaEventRecord(e0);
  for (int i = 0; i < reps; i++) launch();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
#if DK_SPLIT
  const size_t smem_a = sizeof(SimP), smem_b = sizeof(SimN);
#else
  const size_t smem_a = sizeof(Sim), smem_b = 0;
#endif
  printf("{\"nworld\": %d, \"ms\": %.5f, \"err\": \"%s\", \"smem\": %zu, \"blocks_per_sm\": %d, \"tape\": %d, "
         "\"smem_post\": %zu, \"blocks_per_sm_post\": %d}\n", nw, reps ? ms / reps : 0.f,
         cudaGetErrorString(cudaGetLastError()), smem_a, nblk, (int)use_tape, smem_b, nblk_post);
  std::vector<float> hq((size_t)nw * NQ), hv((size_t)nw * NV), hw((size_t)nw * NV); std::vector<int> hi((size_t)nw * 4);
  cudaMemcpy(hq.data(), oq, hq.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hv.data(), ov, hv.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hw.data(), ow, hw.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hi.data(), oi, hi.size() * 4, cudaMemcpyDeviceToHost);
  snprintf(path, sizeof path, "%s/out_qpos.bin", dir); writef(path, hq.data(), hq.size());
  snprintf(path, sizeof path, "%s/out_qvel.bin", dir); writef(path, hv.data(), hv.size());
  snprintf(path, sizeof path, "%s/out_warm.bin", dir); writef(path, hw.data(), hw.size());
  snprintf(path, sizeof path, "%s/out_info.bin", dir); writef(path, hi.data(), hi.size());
  std::vector<float> hc((size_t)nw * MAXCON * 8);
  cudaMemcpy(hc.data(), oc, hc.size() * 4, cudaMemcpyDeviceToHost);
  snprintf(path, sizeof path, "%s/out_con.bin", dir); writef(path, hc.data(), hc.size());
  if (use_tape) {
    std::vector<float> ht(ntape);
    cudaMemcpy(ht.data(), tp, ntape * 4, cudaMemcpyDeviceToHost);
    snprintf(path, sizeof path, "%s/tape.bin", dir); writef(path, ht.data(), ht.size());
  }
  return 0;
}
"""


def contacts_vs_mjw(m, con, ok, ref, K, tol_dist=1e-5, tol_pos=1e-5, tol_normal=1e-4, tol_flip=1e-6):
    """Per world, our contact set (pair, dist, pos, normal) against MuJoCo Warp's (fixtures.load_contacts), matched per
    pair (by count, then nearest position): the share of supported worlds that match within the tolerances, the largest deviations over the
    matched contacts, and the mismatching pairs by geom-type combination. A contact that only one side has and whose
    |dist| < tol_flip is a threshold flip (float32 rounding decides whether it is active), counted separately."""
    names = {int(t): mujoco.mjtGeom(t).name[7:].lower() for t in set(m.geom_type)}
    kind = lambda pr: "-".join(sorted((names[int(m.geom_type[m.pair_geom1[pr]])],
                                       names[int(m.geom_type[m.pair_geom2[pr]])])))
    good, flips, n, dev, bad_kind = 0, 0, 0, np.zeros(3), {}
    for i in range(K):
        if not ok[i]:
            continue
        n += 1
        bypair = {}
        for side, rows in ((0, con[i]), (1, ref["con"][i])):
            for r in rows:
                if r[0] >= 0:
                    bypair.setdefault(int(r[0]), ([], []))[side].append(r)
        same, flip = True, False
        for pr, (a, b) in bypair.items():
            a, b = sorted(a, key=lambda r: r[1]), sorted(b, key=lambda r: r[1])
            k = min(len(a), len(b))
            extra = a[k:] + b[k:]
            if extra and all(abs(r[1]) < tol_flip for r in extra):
                flip = True
                a, b = a[:k], b[:k]
            if len(a) != len(b):
                same = False
                bad_kind[kind(pr)] = bad_kind.get(kind(pr), 0) + 1
                continue
            rest = list(b)
            for x in a:
                y = rest.pop(int(np.argmin([np.abs(x[2:5] - r[2:5]).max() for r in rest])))
                d = np.array([abs(x[1] - y[1]), np.abs(x[2:5] - y[2:5]).max(), abs(abs(np.dot(x[5:8], y[5:8])) - 1)])
                if d[0] < tol_dist and d[1] < tol_pos and d[2] < tol_normal:
                    dev = np.maximum(dev, d)
                else:
                    same = False
                    bad_kind[kind(pr)] = bad_kind.get(kind(pr), 0) + 1
        good += same
        flips += flip and same
    return dict(ours_vs_mjw_contacts=dict(match=good / max(n, 1), worlds=n, threshold_flip_worlds=flips,
                                          max_dist_dev=float(dev[0]), max_pos_dev=float(dev[1]),
                                          max_normal_dev=float(dev[2]), mismatch_kinds=bad_kind))


def ulp_dist(a, b):
    """|a - b| in float32 units in the last place (on the ordered integer representation)."""
    def ordered(x):
        i = np.ascontiguousarray(x, np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    return np.abs(ordered(a) - ordered(b))


def stats(e):
    e = np.asarray(e, np.float64)
    if not e.size:
        return dict(n=0)
    return dict(n=int(e.size), median=float(np.median(e)), p99=float(np.quantile(e, 0.99)), max=float(e.max()))


def engine_frame(n, axis=None):
    """The engine's contact frame rows (n, b, n x b) in float64 (step.py): for plane-capsule b is the capsule axis
    projected on the plane (the fixed basis vector when that projection is shorter than 0.5), else make_frame(n)."""
    n = np.asarray(n, np.float64)
    e = np.array([0., 1., 0.]) if -0.5 < n[1] < 0.5 else np.array([0., 0., 1.])
    if axis is not None:
        b = axis - n * (n @ axis)
        bn = np.linalg.norm(b)
        b = b / bn if bn >= 0.5 else e
    else:
        b = e - n * (n @ e)
        b = b / np.linalg.norm(b)
    return np.stack([n, b, np.cross(n, b)])


def kbimp64(solref, solimp, pos, h):
    """step.py kbimp() in float64: (k, b, imp) with refsafe."""
    tc, dr = max(solref[0], 2 * h), solref[1]
    dmin, dmax = np.clip(solimp[0], 1e-4, 0.9999), np.clip(solimp[1], 1e-4, 0.9999)
    width, mid, p = max(1e-15, solimp[2]), np.clip(solimp[3], 1e-4, 0.9999), max(1.0, solimp[4])
    k, b, x = 1 / max(1e-15, dmax ** 2 * tc ** 2 * dr ** 2), 2 / max(1e-15, dmax * tc), abs(pos) / width
    if dmin == dmax or solimp[2] <= 1e-15:
        imp = 0.5 * (dmin + dmax)
    elif x <= 0 or x >= 1:
        imp = dmin if x <= 0 else dmax
    elif p == 1:
        imp = dmin + x * (dmax - dmin)
    elif x <= mid:
        imp = dmin + x ** p / mid ** (p - 1) * (dmax - dmin)
    else:
        imp = dmin + (1 - (1 - x) ** p / (1 - mid) ** (p - 1)) * (dmax - dmin)
    return k, b, float(np.clip(imp, dmin, dmax))


def tape_checks(args, m, mt, arr, run, main, K):
    """Forward-tape checks T1-T8 on the main run's tape."""
    nw, nsub, nv = len(main["qpos"]), args.substeps, m.nv
    rec = tape.decode(main["tape"], m)
    forms = tape.factor_forms(m)
    keys = ("qpos", "qvel", "warm", "info", "con")
    out = {}

    off = [run(nsub)[1] for _ in range(2)]
    det = all(np.array_equal(off[0][k], off[1][k]) for k in keys)
    same = {k: bool(np.array_equal(main[k], off[0][k])) for k in keys}
    t1 = dict(deterministic_run_to_run=det, tape_on_vs_off_bitwise=same, **{"pass": all(same.values())})
    if not det:
        def rel(a, b):
            return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-9)
        ok = np.array([sorted(x[x[:, 0] >= 0, 0]) == sorted(y[y[:, 0] >= 0, 0])
                       for x, y in zip(main["con"], off[0]["con"])])
        ok &= (rel(main["qpos"], off[0]["qpos"]) <= 1e-6) & (rel(main["qvel"], off[0]["qvel"]) <= 1e-6)
        t1.update(fallback_share=float(ok.mean()), **{"pass": bool(ok.all())})
    out["T1_outputs_tape_on_vs_off"] = t1

    sub = {k: run(k)[1] for k in range(1, nsub)}
    q32, v32 = arr["qpos"].astype(np.float32), arr["qvel"].astype(np.float32)
    t2 = dict(rec0_is_input=bool(np.array_equal(rec["qpos"][:, 0], q32) and np.array_equal(rec["qvel"][:, 0], v32)))
    t2["rec_k_vs_run_of_k_substeps"] = {
        f"k{k}": dict(qpos=bool(np.array_equal(rec["qpos"][:, k], sub[k]["qpos"])),
                      qvel=bool(np.array_equal(rec["qvel"][:, k], sub[k]["qvel"])),
                      qacc_of_k_minus_1_vs_warm=bool(np.array_equal(rec["qacc"][:, k - 1], sub[k]["warm"])))
        for k in sub}
    t2["last_qacc_is_warm_output"] = bool(np.array_equal(rec["qacc"][:, -1], main["warm"]))
    h = np.float32(m.opt.timestep)
    nxt = np.concatenate([rec["qvel"][:, 1:], main["qvel"][:, None]], 1)
    fma = (rec["qvel"].astype(np.float64) + np.float64(h) * rec["qacc_adv"].astype(np.float64)).astype(np.float32)
    u_fma, u_sep = ulp_dist(nxt, fma), ulp_dist(nxt, rec["qvel"] + h * rec["qacc_adv"])
    t2["qvel_chain_ulp"] = dict(fused_max=int(u_fma.max()), fused_share_exact=float((u_fma == 0).mean()),
                                separately_rounded_max=int(u_sep.max()))
    t2["pass"] = bool(t2["rec0_is_input"] and t2["last_qacc_is_warm_output"] and u_fma.max() <= 2
                      and all(all(v.values()) for v in t2["rec_k_vs_run_of_k_substeps"].values()))
    out["T2_state_chain"] = t2

    ptype, pdim = [], np.asarray(m.pair_dim)
    for p in range(m.npair):
        ts = sorted(step.GEOM_TYPE[int(m.geom_type[g])] for g in (m.pair_geom1[p], m.pair_geom2[p]))
        ptype.append(tuple(ts))
    allowed = {(0, 1): 1, (0, 2): 2, (1, 1): 1, (1, 2): 1, (2, 2): 4, (0, 3): 8, (2, 3): 2, (3, 3): 8}
    valid = rec["pair"] >= 0
    pr, sl = rec["pair"][valid], rec["slot"][valid]
    lim = np.array([allowed[ptype[p]] for p in pr])
    dup = sum(len(set(zip(rec["pair"][i, k][valid[i, k]], rec["slot"][i, k][valid[i, k]]))) != valid[i, k].sum()
              for i in range(nw) for k in range(nsub))
    nfl = int((m.dof_frictionloss > 0).sum())
    nrows = np.where(valid, np.where(pdim[np.maximum(rec["pair"], 0)] == 3, 4, 1), 0).sum(-1) + rec["nlim"] + nfl
    c1 = sub[1]["con"] if nsub > 1 else main["con"]
    p1 = c1[:, :, 0].astype(np.int64)
    t3 = dict(rec0_vs_1substep_run_order=bool(np.array_equal(rec["pair"][:, 0], p1)),
              last_vs_out_con_order=bool(np.array_equal(rec["pair"][:, -1], main["con"][:, :, 0].astype(np.int64))),
              rec_k_vs_run_of_k_plus_1_order={f"k{k}": bool(np.array_equal(rec["pair"][:, k], sub[k + 1]["con"][:, :, 0]
                                                                          .astype(np.int64))) for k in range(nsub - 1)},
              slots_invalid=int((sl >= lim).sum()),
              slots=dict(zip(*(x.tolist() for x in np.unique(sl, return_counts=True)))),
              duplicate_pair_slot_records=int(dup),
              nefc_vs_rows=float((rec["nefc"] == np.minimum(nrows, step.NEFC)).mean()),
              ncon_last_vs_info=float((rec["ncon"][:, -1] == np.minimum(main["info"][:, 1], step.maxcon(m))).mean()),
              nefc_last_vs_info=float((rec["nefc"][:, -1] == main["info"][:, 2]).mean()))
    if fixtures.has_contacts(args.corpus):
        ref = fixtures.load_contacts(args.corpus)["con"]
        good = flips = bad = 0
        for i in range(nw):
            if main["info"][i, 3]:
                continue
            a = [(int(r[0]), float(r[1])) for r in c1[i] if r[0] >= 0]
            b = [(int(r[0]), float(r[1])) for r in ref[args.state_index[i]] if r[0] >= 0]
            ca, cb = Counter(p for p, _ in a), Counter(p for p, _ in b)
            if ca == cb:
                good += 1
                continue
            extra = []
            for p in set(ca) | set(cb):
                if ca[p] != cb[p]:
                    da, db = sorted(x for q, x in a if q == p), sorted(x for q, x in b if q == p)
                    n = min(len(da), len(db))
                    extra += da[n:] + db[n:]
            if all(abs(x) < 1e-6 for x in extra):
                flips += 1
            else:
                bad += 1
        t3["vs_mjw_pairs"] = dict(match=good, threshold_flips=flips, mismatch=bad)
    t3["pass"] = bool(t3["rec0_vs_1substep_run_order"] and t3["last_vs_out_con_order"]
                      and all(t3["rec_k_vs_run_of_k_plus_1_order"].values()) and not t3["slots_invalid"] and not dup
                      and t3["nefc_vs_rows"] == 1.0 and t3["ncon_last_vs_info"] == 1.0 and t3["nefc_last_vs_info"] == 1.0
                      and t3.get("vs_mjw_pairs", {}).get("mismatch", 0) == 0)
    out["T3_contact_table"] = t3

    c1c = source.constants(m)[0]
    lay = {s: n for s, n in zip([x[0] for x in source.CONST_LAYOUT], np.cumsum([0] + [
        (dict(nb=m.nbody, nv=m.nv, one=1)[c] if isinstance(c, str) else c) * w for _, _, c, w in source.CONST_LAYOUT]))}
    cget = lambda s, n: c1c[lay[s]:lay[s] + n]
    kp, kv, frc = cget("c_kp", nv), cget("c_kv", nv), cget("c_frclim", nv)
    jr = np.zeros((nv, 2), np.float32)
    jl = np.zeros(nv, bool)
    for j in range(1, m.njnt):
        jr[m.jnt_dofadr[j]], jl[m.jnt_dofadr[j]] = m.jnt_range[j], m.jnt_limited[j]
    qd = rec["qpos"][..., 1:]
    lim_on = (np.arange(nv) >= 6) & jl & (np.minimum(qd - jr[:, 0], jr[:, 1] - qd) < 0)
    ctrl = np.concatenate([np.zeros((nw, 6), np.float32), arr["ctrl"].astype(np.float32)], 1)[:, None]
    act = kp.astype(np.float64) * (ctrl - qd.astype(np.float64)) - kv.astype(np.float64) * rec["qvel"].astype(np.float64)
    cl64 = np.abs(act) >= frc
    cl64[..., :6] = False
    mis = cl64 != rec["clamp_dof"]
    marg = np.abs(np.abs(act) - frc) / frc
    out["clamp_mask"] = dict(dofs_clamped=int(rec["clamp_dof"].sum()), float64_recompute_mismatch=int(mis.sum()),
                             mismatch_max_rel_margin=float(marg[mis].max()) if mis.any() else 0.0,
                             nlim_vs_float32_limit_test=float((lim_on.sum(-1) == rec["nlim"]).mean()))

    d = mujoco.MjData(m)
    torso, foot = fixtures.torso_id(m, mt), mt["foot_pairs"]
    ipos0, fric0 = m.body_ipos[torso].copy(), m.pair_friction.copy()
    kv64 = np.zeros(nv)
    for a in range(m.nu):
        kv64[m.jnt_dofadr[m.actuator_trnid[a, 0]]] = -m.actuator_biasprm[a, 2]
    pair_of = {frozenset((int(m.pair_geom1[i]), int(m.pair_geom2[i]))): i for i in range(m.npair)}
    eng_g = []
    for p in range(m.npair):
        g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        if step.GEOM_TYPE[int(m.geom_type[g1])] > step.GEOM_TYPE[int(m.geom_type[g2])]:
            g1, g2 = g2, g1
        eng_g.append((g1, g2))
    binvw, dinvw, hh = m.body_invweight0[:, 0], m.dof_invweight0, float(np.float32(m.opt.timestep))
    sets = step.pair_param_sets(m)
    LIMIT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
    jp1, jp2 = np.zeros((3, nv)), np.zeros((3, nv))
    e4, e5, e5b, e5c, e5_refact, unmatched, skipped, geo_order = [], [], [], [], [], 0, 0, 0
    worst = {False: [], True: []}
    sing5 = []
    footset = set(int(x) for x in foot)
    for i in range(K):
        m.body_ipos[torso] = ipos0 + arr["com"][i]
        m.pair_friction[:] = fric0
        m.pair_friction[foot, :2] = arr["fric"][i]
        for k in range(nsub):
            d.qpos[:], d.qvel[:], d.ctrl[:] = rec["qpos"][i, k], rec["qvel"][i, k], arr["ctrl"][i]
            mujoco.mj_fwdPosition(m, d)
            M = np.zeros((nv, nv))
            mujoco.mj_fullM(m, d, M)
            Q = M + hh * np.diag(kv64)
            LQ = tape.tri_unpack(rec["L_Q"][i, k], nv).astype(np.float64)
            e4.append(np.linalg.norm(tape.factor_product(LQ, forms["L_Q"]) - Q) / np.linalg.norm(Q))
            if rec["unsupported"][i, k] or nfl:
                skipped += 1
                continue
            nc, zone = int(rec["ncon"][i, k]), rec["zone"][i, k]
            tk = [(int(rec["pair"][i, k, c]), int(rec["slot"][i, k, c])) for c in range(nc)]
            mj = {}
            for ci in range(d.ncon):
                cc = d.contact[ci]
                p = pair_of[frozenset((int(cc.geom1), int(cc.geom2)))]
                slot = 0
                if ptype[p] == (0, 2):
                    g = eng_g[p][1]
                    slot = 0 if (cc.pos - d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3, 3)[:, 2] > 0 else 1
                mj.setdefault((p, slot), []).append(ci)
            mlim = {int(m.jnt_dofadr[d.efc_id[r]]): r for r in range(d.nefc) if d.efc_type[r] == LIMIT}
            ldofs = [j for j in range(nv) if lim_on[i, k, j]]
            if (sorted(mj) != sorted(tk) or any(len(v) != 1 for v in mj.values()) or sorted(mlim) != ldofs):
                unmatched += 1
                con_e = None
            else:
                con_e = True
            ce = (sub[k + 1] if k + 1 < nsub else main)["con"][i]
            Hs = {v: M.copy() for v in (("mujoco", "mujoco_geometry_engine_D") if con_e else ()) + ("engine",)}
            r, dev = 0, np.zeros(3)
            for j in ldofs:
                if zone[r]:
                    q = rec["qpos"][i, k, j + 1]
                    _, _, imp = kbimp64(m.jnt_solref[1], m.jnt_solimp[1], min(q - jr[j, 0], jr[j, 1] - q), hh)
                    for v in Hs:
                        Hs[v][j, j] += d.efc_D[mlim[j]] if v == "mujoco" else 1 / max(dinvw[j] * (1 - imp) / imp, 1e-15)
                r += 1
            for c, (p, slot) in enumerate(tk):
                g1, g2 = eng_g[p]
                b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
                cap = d.geom_xmat[g2].reshape(3, 3)[:, 2] if ptype[p] == (0, 2) else None
                dim, nr = int(pdim[p]), 4 if pdim[p] == 3 else 1
                geo_order += int(ce[c, 0]) != p
                mu = float(np.float32(arr["fric"][i])) if p in footset else sets[0][sets[1][p], 0]
                iw = binvw[b1] + binvw[b2]
                if dim == 3:
                    iw = (iw + mu * mu * iw) * 2 * mu * mu / m.opt.impratio
                _, _, imp = kbimp64(sets[0][sets[1][p], 1:3], sets[0][sets[1][p], 3:8], float(ce[c, 1]), hh)
                De = [1 / max(iw * (1 - imp) / imp, 1e-15)] * nr
                geo = dict(engine=(ce[c, 2:5].astype(np.float64), ce[c, 5:8].astype(np.float64), mu, De))
                if con_e:
                    cc = d.contact[mj[(p, slot)][0]]
                    sg = 1.0 if int(cc.geom1) == g1 else -1.0
                    geo["mujoco"] = (cc.pos.copy(), sg * cc.frame[:3], cc.friction[0],
                                     [d.efc_D[cc.efc_address + t] for t in range(nr)])
                    geo["mujoco_geometry_engine_D"] = geo["mujoco"][:3] + (De,)
                    dev = np.maximum(dev, [abs(float(ce[c, 1]) - cc.dist), np.abs(ce[c, 2:5] - cc.pos).max(),
                                           1 - abs(float(ce[c, 5:8] @ (sg * cc.frame[:3])))])
                for v, (pos, nrm, mu_v, D) in geo.items():
                    mujoco.mj_jac(m, d, jp1, None, pos, b1)
                    mujoco.mj_jac(m, d, jp2, None, pos, b2)
                    J = engine_frame(nrm / np.linalg.norm(nrm), cap) @ (jp2 - jp1)
                    us = ([J[0] + mu_v * J[1], J[0] - mu_v * J[1], J[0] + mu_v * J[2], J[0] - mu_v * J[2]]
                          if dim == 3 else [J[0]])
                    for t in range(nr):
                        if zone[r + t]:
                            Hs[v] += D[t] * np.outer(us[t], us[t])
                r += nr
            LH = tape.tri_unpack(rec["L_H"][i, k], nv).astype(np.float64)
            HL = tape.factor_product(LH, forms["L_H"])
            err = {v: np.linalg.norm(HL - H) / np.linalg.norm(H) for v, H in Hs.items()}
            e5b.append(err["engine"])
            if con_e:
                e5.append(err["mujoco"])
                e5c.append(err["mujoco_geometry_engine_D"])
                sing5.append(bool(rec["singular"][i, k]))
                if rec["refact"][i, k]:
                    e5_refact.append(e5[-1])
                kinds = sorted(Counter("%d-%d" % ptype[p] for p, _ in tk).items())
                w = worst[sing5[-1]]
                worst[sing5[-1]] = sorted(w + [(e5[-1], i, k, e5b[-1], e5c[-1], dev.tolist(), kinds)], reverse=True)[:3]
    m.body_ipos[torso], m.pair_friction[:] = ipos0, fric0
    out["T4_L_Q"] = dict(**stats(e4), **{"pass": bool(np.median(e4) <= 1e-6 and max(e4) <= 1e-5)})
    nrec = K * nsub - skipped
    e5a, s5 = np.asarray(e5), np.asarray(sing5, bool)
    reg = e5a[~s5] if len(e5a) else e5a
    out["T5_L_H"] = dict(**stats(e5), matched_share=float(len(e5) / max(nrec, 1)), unmatched=unmatched,
                         unsupported_skipped=skipped, non_singular=stats(reg),
                         singular=stats(e5a[s5] if len(e5a) else e5a),
                         refact_records=stats(e5_refact), design_threshold_max_1e_5=bool(max(e5, default=0.0) <= 1e-5),
                         **{"pass": None if nrec == 0 else
                            bool(len(e5) >= 0.95 * nrec and len(reg) and np.quantile(reg, 0.99) <= 1e-4
                                 and reg.max() <= 1e-3)})
    out["T5b_L_H_engine_geometry"] = dict(**stats(e5b), contact_order_mismatch=geo_order,
                                          **{"pass": None if nrec == 0 else
                                             bool(geo_order == 0 and max(e5b, default=0.0) <= 1e-5)})
    out["T5c_L_H_mujoco_geometry_engine_D"] = dict(**stats(e5c), non_singular=stats(np.asarray(e5c)[~s5] if len(e5c)
                                                                                      else e5c))
    out["T5_worst"] = {("singular" if sg else "non_singular"): [
        dict(world=i, substep=k, T5=e, T5b=eb, T5c=ec, min_nu=float(rec["min_nu"][i, k]),
             refact=bool(rec["refact"][i, k]), dev_dist_pos_normal=dv, contact_kinds=dict(kd)) for e, i, k, eb, ec, dv, kd in worst[sg]] for sg in worst}

    w_any = lambda x: float(x.any(1).mean())
    t6 = dict(records=int(nw * nsub), refact_records=float(rec["refact"].mean()), refact_worlds=w_any(rec["refact"]),
              nonconv_records=float(rec["nonconv"].mean()), nonconv_worlds=w_any(rec["nonconv"]),
              min_nu_lt_5mm_worlds=int((rec["min_nu"] < 5e-3).any(1).sum()),
              min_nu_lt_1mm_worlds=int((rec["min_nu"] < 1e-3).any(1).sum()),
              self_contact_records=float(np.isfinite(rec["min_nu"]).mean()),
              singular_flag_consistent=bool(np.array_equal(rec["singular"], rec["min_nu"] < tape.SINGULAR_NU)),
              clamp_worlds_substep0=float((rec["clamp"][:, 0] != 0).mean()),
              clamp_records=float((rec["clamp"] != 0).mean()),
              niter_mean_per_substep=rec["niter"].mean(0).round(3).tolist(), niter_max=int(rec["niter"].max()),
              zone_rows_mean=float(rec["zone"].sum(-1).mean()), nefc_mean=float(rec["nefc"].mean()))
    if any("DK_TAPE_VERIFY" in x for x in args.define):
        t6["verify_diff_without_refact"] = int((rec["verify_diff"] & ~rec["refact"]).sum())
        t6["verify_diff_with_refact"] = int((rec["verify_diff"] & rec["refact"]).sum())
    out["T6_statistics"] = t6

    if args.reps > 0:
        ms = {False: [], True: []}
        for _ in range(3):
            for on in (False, True):
                ms[on].append(run(nsub, on, args.reps)[0]["ms"])
        slow = float(np.median(ms[True]) / np.median(ms[False]) - 1)
        out["T7_timing"] = dict(ms_tape_off=ms[False], ms_tape_on=ms[True], slowdown=slow, **{"pass": slow <= 0.04})

    raw = main["tape"].view(np.uint32)
    field = np.zeros(raw.shape[-1], bool)
    for o, n in tape.layout(m).values():
        field[o:o + n] = True
    out["T8_record_coverage"] = dict(unwritten_words=int((raw == 0xA5A5A5A5).sum()),
                                     nonzero_padding_words=int((raw[..., ~field] != 0).sum()),
                                     padding_words_per_record=int((~field).sum()))
    out["T8_record_coverage"]["pass"] = (out["T8_record_coverage"]["unwritten_words"] == 0
                                         and out["T8_record_coverage"]["nonzero_padding_words"] == 0)
    out["record_words"] = tape.rec_words(m)
    out["record_bytes"] = 4 * tape.rec_words(m)
    return out


def forest_modules():
    """(host, step, tape, adjoint) of the forest engine (nablaengine.forest_engine; NotInRelease in a public package)."""
    from nablaengine._optional import FOREST, load
    return load("nablaengine.forest_engine", FOREST, "host", "step", "tape", "adjoint")


def forest_binary_source(m, foot_pairs, com_body):
    """(CUDA source, packed constants) of the engine binary on the forest engine (--engine forest): the forest engine's
    forward and adjoint text as its libraries build it (adjoint.cuda_source, constants adjoint.engine_constants with
    com_body), forest_engine.host's KERNEL and HOST's main (ctrl NU wide, com COM_WORDS, the contact table CON_WORDS;
    a deferred-CCD build also opts k_pre_a into its shared memory). scripts/identity_gate.py --engine forest hashes
    it."""
    fhost, _, _, fadj = forest_modules()
    consts, lays, sizes = fadj.engine_constants(m, foot_pairs, com_body)
    hst = (HOST.replace("IN(ctrl, NV - 6)", "IN(ctrl, NU)").replace("IN(com, 3)", "IN(com, COM_WORDS)")
           .replace("MAXCON * 8", "MAXCON * CON_WORDS"))
    pre = "  cudaFuncSetAttribute(k_pre, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_pre);\n"
    assert hst.count(pre) == 1 and hst.count("MAXCON * CON_WORDS") == 2
    hst = hst.replace(pre, pre + "#if CV_DEFER\n  cudaFuncSetAttribute(k_pre_a, cudaFuncAttributeMaxDynamicSharedMemorySize, "
                                 "(int)smem_pre);\n#endif\n")
    from nablaengine.forest_engine import margin_convex
    src = (fadj.cuda_source(m) + margin_convex.host(fhost.KERNEL, m)
           + hst.replace("LOAD_CONSTS", fhost.loader(lays, sizes)).replace("NCONST", str(len(consts))))
    return src, consts


def binary_source(m, foot_pairs, engine="tree", com_body=None):
    """(CUDA source, packed constants) of this script's engine binary for model m (the forward kernels k_pre / k_post /
    k_substep, KERNEL and HOST's main; scripts/identity_gate.py hashes it). engine "forest": forest_binary_source with
    com_body."""
    if engine == "forest":
        return forest_binary_source(m, foot_pairs, com_body)
    c1, maxd = source.constants(m)
    c2 = step.pair_constants(m, foot_pairs)
    dofbody, mask = step.extra_constants(m)
    halves = np.stack([mask & 0xFFFF, mask >> 16], 1).ravel()
    consts = np.concatenate([c1, c2, dofbody.astype(np.float32), halves.astype(np.float32)])
    sizes = {"nb": m.nbody, "nv": m.nv, "one": 1, "nv2": 2 * m.nv}
    lay1 = source.CONST_LAYOUT
    lay2 = [(s, k, n, 1) for s, k, n in step.const_layout(m)]
    lay3 = [("c_dofbody", "i", "nv", 1), ("c_ancmask", "u", "nb", 1)]
    src = (source.header(m, maxd, 1) + step.EXTRA_HEADER + step.step_header(m) + tape.header(m) + source.DEVICE
           + step.STEP_DEVICE + step.STEP_KERNEL + KERNEL
           + HOST.replace("LOAD_CONSTS", loader(list(lay1) + lay2 + lay3, sizes)).replace("NCONST", str(len(consts))))
    return step.apply_hooks(src, step.tree_chol_hooks(m)), consts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="corpus name (nablaengine.fixtures.corpora(); default mf_s0)")
    ap.add_argument("--select", default="first", choices=("first", "truth"),
                    help="the worlds: the corpus states in order (tiled when --nworld exceeds them), or the states of "
                         "the corpus's truth file first, then the others in order (comparison with the truth)")
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--tolerance", type=float, default=1e-8, help="Newton tolerance of both solvers")
    ap.add_argument("--substeps", type=int, default=1, help="4: a full control step, also compared with the corpus's "
                    "MuJoCo Warp output (50 iterations)")
    ap.add_argument("--check", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--contact-report", type=int, default=0,
                    help="print the contacts of up to N worlds whose contact set differs from MuJoCo's (last substep)")
    ap.add_argument("--xflag", action="append", default=[], help="extra raw nvcc flag, e.g. --xflag=-maxrregcount=128")
    ap.add_argument("--define", action="append", default=[],
                    help="extra preprocessor definition for the engine, e.g. --define DKC=__device__ (repeatable)")
    ap.add_argument("--timeout", type=float, default=900, help="seconds before the engine binary is killed (a hang)")
    ap.add_argument("--blocks-per-sm", type=int, default=0, help=">0: cap the worlds per SM by requesting more shared memory")
    ap.add_argument("--sweep-blocks-per-sm", default="",
                    help="comma-separated caps (0 = no cap): after the check, time the same binary at each (no recompile)")
    ap.add_argument("--fused", action="store_true",
                    help="the fused forward k_substep (one launch per control step; -DDK_SPLIT=0) instead of the split "
                         "kernels k_pre / k_post (two launches per substep, host.dk_forward_split; bitwise the same)")
    ap.add_argument("--split", action="store_true", help="(the default; accepted for compatibility)")
    ap.add_argument("--profile", action="store_true",
                    help="compile with -DDK_PROFILE: per-phase cycles (step.PROF_PHASES) of the first launch")
    ap.add_argument("--tape", action="store_true",
                    help="the main run writes the forward tape (tape.py; tape.bin in the work directory) and the tape "
                         "checks T1-T8 run on it (result key 'tape')")
    ap.add_argument("--save-outputs", default="",
                    help="npz path: all worlds' outputs of the main run (qpos, qvel, warm, info, con; tape with --tape), "
                         "for bitwise comparisons between builds")
    ap.add_argument("--gate", action="store_true", help="apply the pass criteria GATES (exit status 1 on a failure)")
    ap.add_argument("--digests", default="check", choices=("check", "write", "off"),
                    help="the main run's outputs against the manifest of forward digests (nablaengine.checks), or "
                         "write this run's entry to --digests-file")
    ap.add_argument("--digests-file", default="", help="--digests write: the manifest to add the entry to")
    ap.add_argument("--digests-replace", action="store_true",
                    help="--digests write: replace an entry of this key whose digests differ")
    ap.add_argument("--engine", default=None, choices=("tree", "forest"),
                    help="the engine whose build is checked (default: $DK_ENGINE, nablaengine.engines.default: tree, else forest)")
    ap.add_argument("--out", required=True)
    workdir.add_argument(ap)
    args = ap.parse_args()
    workdir.configure(args)
    args.engine = args.engine or engines.default()
    if args.engine == "forest" and (args.tape or args.profile):
        ap.error("--tape / --profile: the tree engine's checks only")
    if args.digests == "write" and not args.digests_file:
        ap.error("--digests write needs --digests-file")
    if args.save_outputs and Path(args.save_outputs).resolve() == Path(args.out).with_suffix(".npz").resolve():
        ap.error("--save-outputs must differ from the check's own npz (--out with suffix .npz)")
    args.corpus = args.corpus or fixtures.default_corpus()
    m, mt = fixtures.load_model()
    m.opt.iterations = args.iterations
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    m.opt.tolerance = args.tolerance
    foot = mt["foot_pairs"]
    com_body = mt.get("torso_body", source.COM_BODY)
    src, consts = binary_source(m, foot, args.engine, com_body)
    estep = forest_modules()[1] if args.engine == "forest" else step
    maxcon, con_words = estep.maxcon(m), (int(estep.CON_WORDS) if args.engine == "forest" else 8)
    work = workdir.work_dir("engine_step_")
    (work / "step.cu").write_text(src)
    extra = ((["-DDK_PROFILE"] if args.profile else []) + [f"-D{d}" for d in args.define] + args.xflag
             + (["-DDK_SPLIT=0"] if args.fused else []))
    compile_cuda(work / "step.cu", work / "step", extra, unit="forward")
    nvcc_extra = list(extra)
    consts.tofile(work / "consts.bin")
    z = fixtures.load_corpus(args.corpus)
    nw, S = args.nworld, len(z["qpos"])
    truth = None
    if args.select == "truth":
        if nw > S:
            ap.error(f"--select truth: --nworld {nw} exceeds the corpus's {S} states")
        truth = fixtures.load_truth(args.corpus)
        first = np.asarray(truth["index"], np.int64)
        idx = np.concatenate([first, np.setdiff1d(np.arange(S), first)])[:nw]
    else:
        idx = np.arange(nw) % S
    args.state_index = idx
    arr = {k: np.asarray(z[src_k])[idx] for k, src_k in
           (("qpos", "qpos"), ("qvel", "qvel"), ("warm", "qacc_warmstart"), ("ctrl", "ctrl"), ("fric", "friction"),
            ("com", "com"))}
    for k, a in arr.items():
        a.astype(np.float32).tofile(work / f"{k}.bin")
    rec_words = tape.rec_words(m)

    def run(nsub, use_tape=False, reps=0, cap=args.blocks_per_sm):
        """One run of the engine binary: (its JSON line, the outputs of all worlds, stdout)."""
        o = subprocess.run([str(work / "step"), str(work), str(nw), str(args.iterations), str(reps), str(nsub),
                            str(cap), str(int(use_tape))], check=True, timeout=args.timeout,
                           stdout=subprocess.PIPE, universal_newlines=True).stdout
        r = json.loads(next(line for line in o.splitlines() if line.startswith('{"nworld"')))
        outs = dict(qpos=np.fromfile(work / "out_qpos.bin", np.float32).reshape(nw, m.nq),
                    qvel=np.fromfile(work / "out_qvel.bin", np.float32).reshape(nw, m.nv),
                    warm=np.fromfile(work / "out_warm.bin", np.float32).reshape(nw, m.nv),
                    info=np.fromfile(work / "out_info.bin", np.int32).reshape(nw, 4),
                    con=np.fromfile(work / "out_con.bin", np.float32).reshape(nw, maxcon, con_words))
        if use_tape:
            outs["tape"] = np.fromfile(work / "tape.bin", np.float32).reshape(nw, nsub, rec_words)
        return r, outs, o

    res, main_out, out = run(args.substeps, args.tape, args.reps)
    if cuda.forward_virtual_arch():
        res["arch_flags"] = cuda.arch_flags("forward")
    if cuda.unit_defines("forward", None, extra):
        res["forward_defines"] = cuda.unit_defines("forward", None, extra)
    if args.save_outputs:
        np.savez(args.save_outputs, **main_out)
    sweep = []
    for cap in (int(c) for c in args.sweep_blocks_per_sm.split(",") if c.strip()):
        r = run(args.substeps, False, max(args.reps, 10), cap)[0]
        sweep.append(dict(cap=cap, blocks_per_sm=r["blocks_per_sm"], ms=r["ms"],
                          **({} if args.fused else {"blocks_per_sm_post": r["blocks_per_sm_post"]})))
    if sweep:
        res["blocks_per_sm_sweep"] = sweep
    prof = [json.loads(line) for line in out.splitlines() if line.startswith('{"prof_cycles"')]
    if prof:
        cyc = np.asarray(prof[0]["prof_cycles"], np.float64)
        part = cyc[:step.PROF_PARTITION]
        res["profile"] = dict(cycles_per_world_substep=dict(zip(step.PROF_PHASES, (cyc / (nw * args.substeps)).round(1).tolist())),
                              share=dict(zip(step.PROF_PHASES, (cyc / part.sum()).round(4).tolist())),
                              total_cycles_per_world_substep=float(part.sum() / (nw * args.substeps)))
    oq, ov, ow, info, con = (main_out[k] for k in ("qpos", "qvel", "warm", "info", "con"))
    con = con[:, :, :8]
    res["engine"] = args.engine
    pair_of = {frozenset((int(m.pair_geom1[i]), int(m.pair_geom2[i]))): i for i in range(m.npair)}
    con_ref = []
    K = min(args.check, nw)
    d = mujoco.MjData(m)
    m.opt.iterations = args.iterations
    torso = fixtures.torso_id(m, mt)
    ipos0, fric0 = m.body_ipos[torso].copy(), m.pair_friction.copy()
    rq, rv, ra, ncon_ref = np.zeros((K, m.nq)), np.zeros((K, m.nv)), np.zeros((K, m.nv)), np.zeros(K, int)
    niter_ref, nefc_ref, nlim_ref = np.zeros(K, int), np.zeros(K, int), np.zeros(K, int)
    for i in range(K):
        m.body_ipos[torso] = ipos0 + arr["com"][i]
        m.pair_friction[:] = fric0
        m.pair_friction[foot, :2] = arr["fric"][i]
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.qacc_warmstart[:], d.ctrl[:] = arr["qpos"][i], arr["qvel"][i], arr["warm"][i], arr["ctrl"][i]
        for _ in range(args.substeps):
            mujoco.mj_step(m, d)
        rq[i], rv[i], ra[i], ncon_ref[i] = d.qpos, d.qvel, d.qacc, d.ncon
        con_ref.append(sorted((pair_of[frozenset((int(c.geom1), int(c.geom2)))], float(c.dist), *c.pos, *c.frame[:3])
                              for c in d.contact[:d.ncon]))
        niter_ref[i], nefc_ref[i] = d.solver_niter[0], d.nefc
        nlim_ref[i] = int((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT).sum())
    m.body_ipos[torso], m.pair_friction[:] = ipos0, fric0
    ok = info[:K, 3] == 0

    def rel(a, b):
        e = np.linalg.norm(a[ok] - b[ok], axis=1) / np.maximum(np.linalg.norm(b[ok], axis=1), 1e-9)
        return dict(median=float(np.median(e)), p99=float(np.quantile(e, 0.99)), max=float(e.max()))

    has_ref = args.substeps == 4 and "out_qpos" in z
    if args.substeps == 4 and not has_ref:
        res["ours_vs_mjw"] = f"not evaluated: no MuJoCo Warp reference of {args.corpus}"
    if has_ref:
        gq, gv = (np.asarray(z[k])[idx[:K]] for k in ("out_qpos", "out_qvel"))
        res["mjw_reference"] = dict(iterations=int(z.get("iterations", 50)), file=(
            f"{args.corpus}_fwd.npz" if fixtures.is_fixture(args.corpus)
            else provenance.rel(fixtures.reference_path(args.corpus))))
        res.update(ours_vs_mjw_qvel=rel(ov[:K], gv), mujoco_vs_mjw_qvel=rel(rv, gv), ours_vs_mjw_qpos=rel(oq[:K], gq),
                   mujoco_vs_mjw_qpos=rel(rq, gq))
        axis_d = fixtures.self_contact_axis_distance(m, arr["qpos"][:K])
        for tag, sel in (("axis_lt_5mm", axis_d < 5e-3), ("rest", axis_d >= 5e-3)):
            sel = sel & ok
            if sel.any():
                e = np.linalg.norm(ov[:K][sel] - gv[sel], axis=1) / np.maximum(np.linalg.norm(gv[sel], axis=1), 1e-9)
                res[f"ours_vs_mjw_qvel_{tag}"] = dict(n=int(sel.sum()), median=float(np.median(e)),
                                                      p99=float(np.quantile(e, 0.99)), max=float(e.max()))
    res.update(supported_share=float(ok.mean()), qacc=rel(ow[:K], ra), qvel=rel(ov[:K], rv), qpos=rel(oq[:K], rq),
               ncon_match=float((info[:K, 1][ok] == ncon_ref[ok]).mean()),
               niter=dict(mean=float(info[:K, 0][ok].mean()), max=int(info[:K, 0][ok].max())),
               nefc_mean=float(info[:K, 2].mean()), ncon_mean=float(info[:K, 1].mean()))
    if truth is not None and args.substeps == 4:
        n = min(len(truth["index"]), nw)
        nk = min(n, K)
        tout = truth["out"][:n]
        mo = truth["model_flags"][idx[:n]] == 0
        sup = info[:n, 3] == 0
        res["truth"] = dict(file=provenance.rel(fixtures.truth_path(args.corpus)), states=int(n),
                            model_ok=int(mo.sum()), unsupported=int((~sup).sum()))
        for name, use in (("all", sup), ("model_ok", sup & mo)):
            for o, ours, mj, sl in (("qvel", ov, rv, slice(m.nq, None)), ("qpos", oq, rq, slice(0, m.nq))):
                res[f"ours_vs_truth_{o}_{name}"] = stats(rel_rows(ours[:n][use], tout[use, sl]))
                res[f"mujoco_vs_truth_{o}_{name}"] = stats(rel_rows(mj[:nk][use[:nk]], tout[:nk][use[:nk], sl]))
    err = np.linalg.norm(ow[:K] - ra, axis=1) / np.maximum(np.linalg.norm(ra, axis=1), 1e-9)
    extra = {}
    if has_ref:
        extra["err_mjw_qvel"] = np.linalg.norm(ov[:K] - gv, axis=1) / np.maximum(np.linalg.norm(gv, axis=1), 1e-9)
    np.savez(Path(args.out).with_suffix(".npz"), err=err, ok=ok, info=info[:K], niter_ref=niter_ref, nefc_ref=nefc_ref,
             nlim_ref=nlim_ref, ncon_ref=ncon_ref, qacc_ref=ra, qacc=ow[:K], con=con[:K], **extra)
    bad, pair_bad = 0, {}
    for i in range(K):
        if not ok[i]:
            continue
        ours = sorted(tuple(float(x) for x in c) for c in con[i] if c[0] >= 0)
        same = len(ours) == len(con_ref[i]) and all(
            int(a[0]) == b[0] and abs(a[1] - b[1]) < 1e-4 and np.abs(np.subtract(a[2:5], b[2:5])).max() < 1e-4
            for a, b in zip(ours, con_ref[i]))
        if same:
            continue
        pairs = {int(c[0]) for c in ours} ^ {c[0] for c in con_ref[i]} or {int(c[0]) for c in ours}
        for pr in pairs:
            key = (int(m.geom_type[m.pair_geom1[pr]]), int(m.geom_type[m.pair_geom2[pr]]), pr)
            pair_bad[key] = pair_bad.get(key, 0) + 1
        if bad < args.contact_report:
            print(f"world {i}: ours {len(ours)} contacts, MuJoCo {len(con_ref[i])}")
            for c in ours:
                print("  ours  pair %2d dist %+.5f pos %s n %s" % (int(c[0]), c[1], np.round(c[2:5], 4), np.round(c[5:8], 3)))
            for c in con_ref[i]:
                print("  mjc   pair %2d dist %+.5f pos %s n %s" % (c[0], c[1], np.round(c[2:5], 4), np.round(c[5:8], 3)))
        bad += 1
    res.update(contact_set_mismatch=bad / max(int(ok.sum()), 1), mismatch_pairs={f"{k[0]}-{k[1]} pair {k[2]}": v for k, v in
                                                              sorted(pair_bad.items(), key=lambda x: -x[1])})
    if args.substeps == 1 and fixtures.has_contacts(args.corpus):
        ref = dict(con=fixtures.load_contacts(args.corpus)["con"][idx[:K]])
        res.update(contacts_vs_mjw(m, con[:K], ok, ref, min(K, S)))
    if args.tape:
        res["tape"] = tape_checks(args, m, mt, arr, run, main_out, K)
    if args.digests != "off":
        run_key = dict(substeps=args.substeps, iterations=args.iterations, tolerance=args.tolerance, nworld=nw,
                       select=args.select, tape=bool(args.tape), fused=bool(args.fused))
        if args.engine == "forest":
            run_key["engine"] = "forest"
        key = checks.forward_key(args.corpus, arr, fixtures.fixtures_dir() / "g1_train.mjb", run_key, nvcc_extra)
        if args.digests == "write":
            status = checks.digest_write(args.digests_file, main_out, key, hashlib.sha256(src.encode()).hexdigest(),
                                         replace=args.digests_replace)
            res["digests"] = dict(status=status, manifest=provenance.rel(args.digests_file))
        else:
            res["digests"] = checks.digest_check(main_out, key)
        res["digests"]["key"] = key
    res["failures"] = failures(args, res)
    if args.gate:
        res["gates"] = GATES
    print(json.dumps(res, indent=1), flush=True)
    res["provenance"] = provenance.provenance()
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"engine step check ({args.engine} engine, {args.corpus}, {args.substeps} substeps, {nw} worlds): "
          + ("PASS" if not res["failures"] else "FAIL: " + "; ".join(res["failures"])), flush=True)
    return 1 if res["failures"] else 0


def rel_rows(a, b):
    """Per row: |a - b| / max(|b|, 1e-9) (float64)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-9)


def failures(args, res):
    """The failed checks of a result: the tape checks (TAPE_GATED), the digests (mismatch), and with --gate GATES."""
    out = []
    for k in TAPE_GATED if isinstance(res.get("tape"), dict) else ():
        if res["tape"].get(k, {}).get("pass") is False:
            out.append(f"tape {k}")
    if res.get("digests", {}).get("status") == "mismatch":
        out.append(f"digests mismatch {res['digests'].get('arrays')}")
    if not args.gate:
        return out
    g = GATES

    def le(name, value, bound):
        if not value <= bound:
            out.append(f"{name} {value:.3g} > {bound:g}")

    if res["supported_share"] < g["supported_share"]:
        out.append(f"supported_share {res['supported_share']}")
    if args.substeps == 4:
        if "ours_vs_mjw_qvel" not in res:
            out.append("no MuJoCo Warp reference")
        elif res["mjw_reference"]["iterations"] == args.iterations:
            for o in ("qvel", "qpos"):
                le(f"ours_vs_mjw_{o} median", res[f"ours_vs_mjw_{o}"]["median"], g[f"mjw_{o}_median"])
                le(f"ours_vs_mjw_{o} p99", res[f"ours_vs_mjw_{o}"]["p99"], g[f"mjw_{o}_p99"])
        if args.select == "truth":
            for o in ("qvel", "qpos"):
                e = res[f"ours_vs_truth_{o}_model_ok"]
                if not e.get("n"):
                    out.append(f"ours_vs_truth_{o}_model_ok: no states")
                    continue
                le(f"ours_vs_truth_{o}_model_ok median", e["median"], g[f"truth_{o}_median"])
                le(f"ours_vs_truth_{o}_model_ok p99", e["p99"], g[f"truth_{o}_p99"])
    if args.substeps == 1:
        c = res.get("ours_vs_mjw_contacts")
        if c is None:
            out.append("no MuJoCo Warp contacts")
        elif c["match"] < g["mjw_contacts_match"]:
            out.append(f"ours_vs_mjw_contacts match {c['match']:.4f} < {g['mjw_contacts_match']}")
    return out


if __name__ == "__main__":
    sys.exit(main())
