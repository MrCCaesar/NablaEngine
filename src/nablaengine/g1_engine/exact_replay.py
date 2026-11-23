"""Exact contact replay for box-type pairs (both engines): the geometry helpers that the forward collision and the
contact adjoint's replay both evaluate, at different call sites, explicitly rounded."""
from nablaengine.g1_engine.replay_guard import primitive_only

QUAT2MAT = """__device__ __forceinline__ void quat2mat(const float* q, float* R) {
  float q00 = q[0] * q[0], q01 = q[0] * q[1], q02 = q[0] * q[2], q03 = q[0] * q[3], q11 = q[1] * q[1];
  float q12 = q[1] * q[2], q13 = q[1] * q[3], q22 = q[2] * q[2], q23 = q[2] * q[3], q33 = q[3] * q[3];
  R[0] = q00 + q11 - q22 - q33; R[1] = 2.f * (q12 - q03); R[2] = 2.f * (q13 + q02);
  R[3] = 2.f * (q12 + q03); R[4] = q00 - q11 + q22 - q33; R[5] = 2.f * (q23 - q01);
  R[6] = 2.f * (q13 - q02); R[7] = 2.f * (q23 + q01); R[8] = q00 - q11 - q22 + q33;
}
"""
QUAT2MAT_RN = """__device__ __forceinline__ void quat2mat(const float* q, float* R) {  // (g1_engine.exact_replay: explicitly
  // rounded, the uncontracted expressions: the forward's and the adjoint replay's call sites give the same bits)
  const float q00 = RMUL(q[0], q[0]), q01 = RMUL(q[0], q[1]), q02 = RMUL(q[0], q[2]), q03 = RMUL(q[0], q[3]);
  const float q11 = RMUL(q[1], q[1]), q12 = RMUL(q[1], q[2]), q13 = RMUL(q[1], q[3]), q22 = RMUL(q[2], q[2]);
  const float q23 = RMUL(q[2], q[3]), q33 = RMUL(q[3], q[3]);
  R[0] = RSUB(RSUB(RADD(q00, q11), q22), q33); R[1] = RMUL(2.f, RSUB(q12, q03)); R[2] = RMUL(2.f, RADD(q13, q02));
  R[3] = RMUL(2.f, RADD(q12, q03)); R[4] = RSUB(RADD(RSUB(q00, q11), q22), q33); R[5] = RMUL(2.f, RSUB(q23, q01));
  R[6] = RMUL(2.f, RSUB(q13, q02)); R[7] = RMUL(2.f, RADD(q23, q01)); R[8] = RADD(RSUB(RSUB(q00, q11), q22), q33);
}
"""

PLANE_BOX_FWD = (
    ("      float fr[9], cd = dot3(p2, n) - dot3(p1, n);\n",
     "      float fr[9], cd = RSUB(dot3_rn(p2, n), dot3_rn(p1, n));  // (exact_replay: rounded, as the adjoint's replay)\n"),
    ("        qrot(q2, cl, cw);\n"
     "        float dist = cd + dot3(n, cw), pos[3];\n"
     "        for (int k = 0; k < 3; k++) pos[k] = cw[k] + p2[k] - 0.5f * n[k] * dist;\n",
     "        qrot_rn(q2, cl, cw);\n"
     "        float dist = RADD(cd, dot3_rn(n, cw)), pos[3];\n"
     "        for (int k = 0; k < 3; k++) pos[k] = RSUB(RADD(cw[k], p2[k]), RMUL(RMUL(0.5f, n[k]), dist));\n"),
)
PLANE_BOX_ADJ = (
    ("      qrot(q2, cl, cw);\n"
     "      float cd = dot3(p2, n) - dot3(p1, n);\n"
     "      *dist = cd + dot3(n, cw);\n"
     "      for (int k = 0; k < 3; k++) pos[k] = cw[k] + p2[k] - 0.5f * n[k] * (*dist);\n",
     "      qrot_rn(q2, cl, cw);  // (exact_replay: the forward's rounded arithmetic)\n"
     "      float cd = RSUB(dot3_rn(p2, n), dot3_rn(p1, n));\n"
     "      *dist = RADD(cd, dot3_rn(n, cw));\n"
     "      for (int k = 0; k < 3; k++) pos[k] = RSUB(RADD(cw[k], p2[k]), RMUL(RMUL(0.5f, n[k]), *dist));\n"),
)


def needed(m):
    """A contact pair with a geom other than plane, sphere and capsule."""
    return not primitive_only(m)


def apply(src, m):
    """src (an engine text: adjoint.cuda_source's, or a forward-only one) with the explicitly rounded helpers where
    needed(m); else unchanged. Each edit applies where its text is present (once; RuntimeError for several copies)."""
    if not needed(m):
        return src
    for a, b in ((QUAT2MAT, QUAT2MAT_RN),) + PLANE_BOX_FWD + PLANE_BOX_ADJ:
        n = src.count(a)
        if n > 1:
            raise RuntimeError(f"exact_replay: {n} copies of {a.strip().splitlines()[0][:60]!r}")
        src = src.replace(a, b)
    return src
