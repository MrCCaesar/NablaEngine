"""Free-body lane spreading in the forest adjoint's contact reduction (adj_contacts)."""
import mujoco
import numpy as np

SETUP_OLD = """\
  const bool body = lane >= 1 && lane < NB;
  const int rb = body ? lane : 1, jt = JT_OF(rb), di = c_dadr[rb];  // the lane's body, its joint and first dof
  float zl[3][3];  // the lane's rotation axes: a free body's R e_k, a hinge / slide's z
  for (int e = 0; e < 3; e++)
    for (int k = 0; k < 3; k++) zl[e][k] = 0.f;
  if (body && jt == JFREE)
    for (int e = 0; e < 3; e++) {
      const float ek[3] = {e == 0 ? 1.f : 0.f, e == 1 ? 1.f : 0.f, e == 2 ? 1.f : 0.f};
      qrot(s.Q[rb], ek, zl[e]);
    }
  else if (body)
    for (int k = 0; k < 3; k++) zl[0][k] = s.z[rb][k];
"""
SETUP_NEW = """\
  const bool body = lane >= 1 && lane < NB;
  // (root_spread) the spare lanes 0, 31, 30, .. in pairs: rotation dofs 1, 2 of the free bodies ADJ_SPREAD_BODY(k)
  const int sx = lane == 0 ? 0 : (lane >= NB ? 32 - lane : 2 * ADJ_SPREAD_N);
  const bool rot = sx < 2 * ADJ_SPREAD_N;  // a helper lane
  const int rb = body ? lane : (rot ? ADJ_SPREAD_BODY(sx >> 1) : 1), jt = JT_OF(rb), di = c_dadr[rb];
  const bool own = body && ((ADJ_SPREAD >> rb) & 1u);  // a spread free body's lane: rotation dof 0
  const int e0 = rot ? 1 + (sx & 1) : 0, nr = rot || own ? 1 : 3;  // a free joint's first rotation dof, count
  float zl[3][3];  // the lane's rotation axes: a free body's R e_k (k = e0 ..), a hinge / slide's z
  for (int e = 0; e < 3; e++)
    for (int k = 0; k < 3; k++) zl[e][k] = 0.f;
  if ((body || rot) && jt == JFREE)
    for (int e = 0; e < nr; e++) {
      const int x = e0 + e;
      const float ek[3] = {x == 0 ? 1.f : 0.f, x == 1 ? 1.f : 0.f, x == 2 ? 1.f : 0.f};
      qrot(s.Q[rb], ek, zl[e]);
    }
  else if (body)
    for (int k = 0; k < 3; k++) zl[0][k] = s.z[rb][k];
"""
LOOP = (
    ("      if (body && i1 != i2) {  // this lane's dofs in the support (contact_dof_vjp)\n",
     "      if ((body || rot) && i1 != i2) {  // this lane's dofs in the support (contact_dof_vjp)\n"),
    ("        const int nrot = jt == JFREE ? 3 : 1, dr = jt == JFREE ? di + 3 : di;\n",
     "        const int nrot = jt == JFREE ? nr : 1, dr = jt == JFREE ? di + 3 + e0 : di;\n"),
    ("        if (jt == JFREE)\n"
     "          for (int k = 0; k < 3; k++) pt[k] += sg * wv[k];  // the free body's translations: columns e_k, no pose term\n",
     "        if (jt == JFREE && body)\n"
     "          for (int k = 0; k < 3; k++) pt[k] += sg * wv[k];  // the free body's translations: columns e_k, no pose term\n"),
)
END_OLD = """\
    if (body) {
      for (int k = 0; k < 3; k++) { s.fb[rb][k] += f[k]; s.tb[rb][k] += t[k]; }
      if (jt == JFREE)
        for (int k = 0; k < 3; k++) { s.Pv[di + k] += pt[k]; s.Pv[di + 3 + k] += pv[k]; }
      else
        s.Pv[di] += pv[0];
    }
"""
END_NEW = """\
    {  // (root_spread) a spread free body's lane adds its helpers' f bar, tau bar (lanes l1: dof 1, l2: dof 2)
      const int k = __popc(ADJ_SPREAD & ((1u << lane) - 1u));
      const int l1 = own ? (k == 0 ? 0 : 32 - 2 * k) : lane, l2 = own ? 31 - 2 * k : lane;
      float o1[6], o2[6];
      for (int q = 0; q < 3; q++) {
        o1[q] = __shfl_sync(0xffffffffu, f[q], l1); o1[3 + q] = __shfl_sync(0xffffffffu, t[q], l1);
        o2[q] = __shfl_sync(0xffffffffu, f[q], l2); o2[3 + q] = __shfl_sync(0xffffffffu, t[q], l2);
      }
      if (own)
        for (int q = 0; q < 3; q++) { f[q] += o1[q] + o2[q]; t[q] += o1[3 + q] + o2[3 + q]; }
    }
    if (body) {
      for (int k = 0; k < 3; k++) { s.fb[rb][k] += f[k]; s.tb[rb][k] += t[k]; }
      if (jt == JFREE)
        for (int k = 0; k < 3; k++) {
          s.Pv[di + k] += pt[k];
          if (k < nr) s.Pv[di + 3 + k] += pv[k];
        }
      else
        s.Pv[di] += pv[0];
    } else if (rot)
      s.Pv[di + 3 + e0] += pv[0];
"""
FN = "template <bool DBG>\n__device__ __noinline__ void adj_contacts(Adj& s, int lane, float* dbg_in) {\n"


def spread_bodies(m):
    """The free bodies whose rotation dofs 1, 2 go to spare lanes: ascending, one pair of spare lanes (0, 31, 30, ..,
    NB) each while pairs last; [] for 32 or more bodies."""
    free = [b for b in range(1, m.nbody) if m.body_jntnum[b] >= 1
            and int(m.jnt_type[m.body_jntadr[b]]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    spare = 1 + 32 - m.nbody if m.nbody <= 32 else 0
    return free[:spare // 2]


def _macros(bodies):
    sel = str(bodies[-1])
    for k in range(len(bodies) - 2, -1, -1):
        sel = f"((k) == {k} ? {bodies[k]} : {sel})"
    mask = int(np.bitwise_or.reduce([1 << b for b in bodies]))
    return (f"#define ADJ_SPREAD 0x{mask:x}u  // (root_spread) the free bodies with helper lanes\n"
            f"#define ADJ_SPREAD_N {len(bodies)}\n#define ADJ_SPREAD_BODY(k) {sel}\n")


def _rep(src, a, b):
    if src.count(a) != 1:
        raise RuntimeError(f"root_spread anchor found {src.count(a)} times, expected 1: {a[:70]!r}")
    return src.replace(a, b)


def _switch(new, old):
    return "#if DK_ADJ_SPREAD\n" + new + "#else\n" + old + "#endif\n"


def apply(src, m):
    """adjoint.cuda_source's text with the free-body lane spreading in adj_contacts; unchanged without spread bodies."""
    bodies = spread_bodies(m)
    if not bodies:
        return src
    i = src.index(FN)
    j = src.index("\n}\n", i) + 3
    fn = src[i:j]
    fn = _rep(fn, SETUP_OLD, _switch(SETUP_NEW, SETUP_OLD))
    for a, b in LOOP:
        fn = _rep(fn, a, _switch(b, a))
    fn = _rep(fn, END_OLD, _switch(END_NEW, END_OLD))
    head = "#ifndef DK_ADJ_SPREAD\n#define DK_ADJ_SPREAD 1\n#endif\n" + _macros(bodies)
    return src[:i] + head + fn + src[j:]
