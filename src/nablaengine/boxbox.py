"""MuJoCo Warp's box-box primitive as device code over a scalar type (mujoco_warp/_src/collision_primitive_core.py,
box_box and _box_box_clip, the same in MuJoCo Warp 3.17 and 4.0; Copyright 2025 The Newton Developers, Apache License
2.0; itself a port of MuJoCo's mjc_BoxBox): a separating-axis test that prefers face axes on near-ties, then the
incident face clipped against the reference face (up to 8 contacts) or one edge-edge contact. Its order of operations,
so that the contacts agree with MuJoCo Warp's up to float rounding."""

DEVICE = r"""
// MuJoCo Warp's box_box (collision_primitive_core.py; nablaengine.boxbox), over a scalar type
#define BB_SEPEPS 1e-6f    // slack on the separation tests, times the box scale
#define BB_PAREPS 1e-7f    // sin^2 below which an edge-cross axis is noise
#define BB_SGNEPS 1e-5f    // axis component below which a support corner is ambiguous
#define BB_DUPEPS 1e-10f   // squared relative radius for clip-vertex deduplication
#define BB_EDGEBIAS 1e-6f  // relative penalty on edge-axis separation on near-ties with the best face axis
#define BB_MAXVERT 8
#define BB_MAXVAL 1e10f    // mjMAXVAL
#define BB_MINVAL 1e-15f   // mjMINVAL

template <class T>
__device__ __forceinline__ T bb_clamp(T x, float lo, float hi) { return t_fmin(t_fmax(x, lo), hi); }

// the polygon (nin vertices) clipped against the half-plane sign * v[coord] <= limit (_box_box_clip; unchanged when
// every vertex is inside)
template <class T>
__device__ int bb_clip(int nin, T (*poly)[3], int coord, float sign, float limit) {
  T d[BB_MAXVERT];
  bool all_inside = true;
  for (int k = 0; k < nin; k++) {
    d[k] = sign * poly[k][coord] - limit;
    if (!(d[k] <= 0.f)) all_inside = false;
  }
  if (all_inside) return nin;
  T out[BB_MAXVERT][3];
  int nout = 0;
  for (int k = 0; k < nin; k++) {
    const int k1 = k + 1 == nin ? 0 : k + 1;
    const T dp = d[k], dq = d[k1];
    if (dp <= 0.f && nout < BB_MAXVERT) {
      for (int c = 0; c < 3; c++) out[nout][c] = poly[k][c];
      nout++;
    }
    if (((dp < 0.f && dq > 0.f) || (dp > 0.f && dq < 0.f)) && nout < BB_MAXVERT) {
      const T t = dp / (dp - dq);
      for (int c = 0; c < 3; c++) out[nout][c] = poly[k][c] + t * (poly[k1][c] - poly[k][c]);
      nout++;
    }
  }
  for (int k = 0; k < nout; k++)
    for (int c = 0; c < 3; c++) poly[k][c] = out[k][c];
  return nout;
}

template <class T>
__device__ int bb_box_box_t(const T* p1, const T* R1, const float* s1, const T* p2, const T* R2, const float* s2,
                            float margin, T* cdist, T (*cpos)[3], T* cnrm) {
  // rot: box 2's axes in box 1's frame (columns); pos21: box 2's center in box 1's frame; pos12: box 1's in box 2's
  T d21[3], d12[3], pos21[3], pos12[3], rot[3][3], rabs[3][3];
  for (int k = 0; k < 3; k++) { d21[k] = p2[k] - p1[k]; d12[k] = p1[k] - p2[k]; }
  for (int i = 0; i < 3; i++) {
    pos21[i] = R1[i] * d21[0] + R1[3 + i] * d21[1] + R1[6 + i] * d21[2];
    pos12[i] = R2[i] * d12[0] + R2[3 + i] * d12[1] + R2[6 + i] * d12[2];
  }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      rot[i][j] = R1[i] * R2[j] + R1[3 + i] * R2[3 + j] + R1[6 + i] * R2[6 + j];
      rabs[i][j] = t_fabs(rot[i][j]);
    }
  // stage 1: the separating-axis test (with rounding slack)
  const float septol = margin + BB_SEPEPS * (s1[0] + s1[1] + s1[2] + s2[0] + s2[1] + s2[2]);
  T sep_best = T(-BB_MAXVAL);
  int code = -1;
  for (int i = 0; i < 3; i++) {  // face axes of box 1
    T radius2 = rabs[i][0] * s2[0] + rabs[i][1] * s2[1] + rabs[i][2] * s2[2];
    T sep = t_fabs(pos21[i]) - s1[i] - radius2;
    if (sep > septol) return 0;
    if (sep > sep_best) { sep_best = sep; code = i; }
  }
  for (int j = 0; j < 3; j++) {  // face axes of box 2
    T radius1 = rabs[0][j] * s1[0] + rabs[1][j] * s1[1] + rabs[2][j] * s1[2];
    T sep = t_fabs(pos12[j]) - s2[j] - radius1;
    if (sep > septol) return 0;
    if (sep > sep_best) { sep_best = sep; code = 3 + j; }
  }
  const T sep_face = sep_best;
  const int code_face = code;
  for (int i = 0; i < 3; i++)  // edge-cross axes: axis i of box 1 x axis j of box 2
    for (int j = 0; j < 3; j++) {
      const int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
      T ax1 = -rot[i2][j], ax2 = rot[i1][j];
      T norm2 = ax1 * ax1 + ax2 * ax2;
      if (norm2 < BB_PAREPS) continue;
      T inv = 1.f / t_sqrt(norm2);
      ax1 = ax1 * inv;
      ax2 = ax2 * inv;
      T radius1 = s1[i1] * t_fabs(ax1) + s1[i2] * t_fabs(ax2);
      const int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      T a21 = ax1 * rot[i1][j1] + ax2 * rot[i2][j1];
      T a22 = ax1 * rot[i1][j2] + ax2 * rot[i2][j2];
      T radius2 = s2[j1] * t_fabs(a21) + s2[j2] * t_fabs(a22);
      T sep = t_fabs(ax1 * pos21[i1] + ax2 * pos21[i2]) - radius1 - radius2;
      if (sep > septol) return 0;
      if (sep - BB_EDGEBIAS * t_fabs(sep) > sep_best && sep > sep_face) { sep_best = sep; code = 6 + 3 * i + j; }
    }
  if (code < 0) return 0;
  int edge_i = 0, edge_j = 0, i1 = 0, i2 = 0;
  T axis[3] = {T(0.f), T(0.f), T(0.f)};
  if (code >= 6) {  // an edge axis within ~8 degrees of the best face axis yields to it unless 5% deeper
    edge_i = (code - 6) / 3;
    edge_j = (code - 6) % 3;
    i1 = (edge_i + 1) % 3;
    i2 = (edge_i + 2) % 3;
    axis[i1] = -rot[i2][edge_j];
    axis[i2] = rot[i1][edge_j];
    T len = t_sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (len > 0.f) for (int k = 0; k < 3; k++) axis[k] = axis[k] / len;
    T face_dot;
    if (code_face < 3) {
      face_dot = t_fabs(axis[code_face]);
    } else {
      const int f = code_face - 3;
      face_dot = t_fabs(axis[0] * rot[0][f] + axis[1] * rot[1][f] + axis[2] * rot[2][f]);
    }
    if (face_dot > 0.99f && sep_best < sep_face + 0.05f * t_fabs(sep_face) + BB_MINVAL) {
      code = code_face;
      sep_best = sep_face;
    }
  }
  if (code >= 6) {  // stage 2a: one edge-edge contact
    const int j1 = (edge_j + 1) % 3, j2 = (edge_j + 2) % 3;
    if (axis[0] * pos21[0] + axis[1] * pos21[1] + axis[2] * pos21[2] < 0.f)
      for (int k = 0; k < 3; k++) axis[k] = -axis[k];
    T a2[3];
    for (int c = 0; c < 3; c++) a2[c] = axis[0] * rot[0][c] + axis[1] * rot[1][c] + axis[2] * rot[2][c];
    int amb1 = -1, amb2 = -1;
    if (t_fabs(axis[i1]) < BB_SGNEPS) amb1 = i1;
    else if (t_fabs(axis[i2]) < BB_SGNEPS) amb1 = i2;
    if (t_fabs(a2[j1]) < BB_SGNEPS) amb2 = j1;
    else if (t_fabs(a2[j2]) < BB_SGNEPS) amb2 = j2;
    T d2[3] = {rot[0][edge_j], rot[1][edge_j], rot[2][edge_j]};
    T b = d2[edge_i];
    T denom = 1.f - b * b;
    T w1[3] = {T(0.f), T(0.f), T(0.f)}, w2[3] = {T(0.f), T(0.f), T(0.f)};
    T best_d2 = T(BB_MAXVAL);
    for (int v1 = 0; v1 < (amb1 >= 0 ? 2 : 1); v1++)
      for (int v2 = 0; v2 < (amb2 >= 0 ? 2 : 1); v2++) {
        T c1[3] = {T(0.f), T(0.f), T(0.f)}, cc[3] = {T(0.f), T(0.f), T(0.f)}, c2[3];
        c1[i1] = T(axis[i1] >= 0.f ? s1[i1] : -s1[i1]);
        c1[i2] = T(axis[i2] >= 0.f ? s1[i2] : -s1[i2]);
        if (amb1 >= 0 && v1 != 0) c1[amb1] = -c1[amb1];
        cc[j1] = T(a2[j1] >= 0.f ? -s2[j1] : s2[j1]);
        cc[j2] = T(a2[j2] >= 0.f ? -s2[j2] : s2[j2]);
        if (amb2 >= 0 && v2 != 0) cc[amb2] = -cc[amb2];
        for (int r = 0; r < 3; r++) c2[r] = rot[r][0] * cc[0] + rot[r][1] * cc[1] + rot[r][2] * cc[2] + pos21[r];
        T e[3] = {c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]};
        T d1e = e[edge_i];
        T d2e = d2[0] * e[0] + d2[1] * e[1] + d2[2] * e[2];
        T s = T(0.f);
        if (denom >= BB_MINVAL) s = (d1e - b * d2e) / denom;
        s = bb_clamp(s, -s1[edge_i], s1[edge_i]);
        T t = bb_clamp(b * s - d2e, -s2[edge_j], s2[edge_j]);
        s = bb_clamp(d1e + b * t, -s1[edge_i], s1[edge_i]);
        T q1[3] = {c1[0], c1[1], c1[2]}, q2[3], g[3];
        q1[edge_i] = s;
        for (int k = 0; k < 3; k++) { q2[k] = c2[k] + d2[k] * t; g[k] = q2[k] - q1[k]; }
        T gap2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if (gap2 < best_d2) {
          best_d2 = gap2;
          for (int k = 0; k < 3; k++) { w1[k] = q1[k]; w2[k] = q2[k]; }
        }
      }
    T dw[3] = {w2[0] - w1[0], w2[1] - w1[1], w2[2] - w1[2]};
    T dist = dw[0] * axis[0] + dw[1] * axis[1] + dw[2] * axis[2];
    if (dist > septol) return 0;
    T mid[3];
    for (int k = 0; k < 3; k++) mid[k] = 0.5f * (w1[k] + w2[k]);
    cdist[0] = dist;
    for (int r = 0; r < 3; r++) {
      cpos[0][r] = R1[3 * r] * mid[0] + R1[3 * r + 1] * mid[1] + R1[3 * r + 2] * mid[2] + p1[r];
      cnrm[r] = R1[3 * r] * axis[0] + R1[3 * r + 1] * axis[1] + R1[3 * r + 2] * axis[2];
    }
    return 1;
  }
  // stage 2b: the incident face clipped against the reference face (the owner of the winning face axis)
  const bool ref1 = code < 3;
  const int a = ref1 ? code : code - 3;
  const float* sref = ref1 ? s1 : s2;
  const float* sinc = ref1 ? s2 : s1;
  const T* pref = ref1 ? p1 : p2;
  const T* mref = ref1 ? R1 : R2;
  const T* posoi = ref1 ? pos21 : pos12;
  T rinc[3][3];  // the incident box's axes in the reference frame (columns)
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) rinc[r][c] = ref1 ? rot[r][c] : rot[c][r];
  const float sgn = posoi[a] >= 0.f ? 1.f : -1.f;
  int binc = 0;
  for (int k = 1; k < 3; k++)
    if (t_fabs(rinc[a][k]) > t_fabs(rinc[a][binc])) binc = k;
  const float tinc = sgn * rinc[a][binc] > 0.f ? -1.f : 1.f;
  const int ax = (a + 1) % 3, ay = (a + 2) % 3, bu = (binc + 1) % 3, bv = (binc + 2) % 3;
  const int coords[3] = {ax, ay, a};
  T cx[3], du[3], dv[3];
  for (int r = 0; r < 3; r++) {
    const int c = coords[r];
    cx[r] = posoi[c] + tinc * sinc[binc] * rinc[c][binc];
    du[r] = sinc[bu] * rinc[c][bu];
    dv[r] = sinc[bv] * rinc[c][bv];
  }
  cx[2] = sgn * cx[2] - sref[a];
  du[2] = du[2] * sgn;
  dv[2] = dv[2] * sgn;
  T poly[BB_MAXVERT][3];
  for (int k = 0; k < 4; k++) {
    const float su = (k == 0 || k == 3) ? 1.f : -1.f, sv = k < 2 ? 1.f : -1.f;
    for (int r = 0; r < 3; r++) poly[k][r] = cx[r] + su * du[r] + sv * dv[r];
  }
  int nvert = 4;
  nvert = bb_clip(nvert, poly, 0, 1.f, sref[ax]);
  nvert = bb_clip(nvert, poly, 0, -1.f, sref[ax]);
  nvert = bb_clip(nvert, poly, 1, 1.f, sref[ay]);
  nvert = bb_clip(nvert, poly, 1, -1.f, sref[ay]);
  int naccept = 0;
  const float dupe2 = BB_DUPEPS * (sref[ax] * sref[ax] + sref[ay] * sref[ay]);
  for (int k = 0; k < nvert; k++) {
    if (poly[k][2] > margin) continue;
    bool dupe = false;
    for (int q = 0; q < naccept; q++) {
      T dx = cpos[q][0] - poly[k][0], dy = cpos[q][1] - poly[k][1];
      if (dx * dx + dy * dy < dupe2) { dupe = true; break; }
    }
    if (!dupe) {
      for (int r = 0; r < 3; r++) cpos[naccept][r] = poly[k][r];
      naccept++;
    }
  }
  if (naccept == 0) return 0;
  const float nsign = ref1 ? sgn : -sgn;
  for (int r = 0; r < 3; r++) cnrm[r] = nsign * mref[3 * r + a];
  for (int k = 0; k < naccept; k++) {
    T v[3] = {cpos[k][0], cpos[k][1], cpos[k][2]}, pc[3];
    pc[ax] = v[0];
    pc[ay] = v[1];
    pc[a] = sgn * (sref[a] + 0.5f * v[2]);
    cdist[k] = v[2];
    for (int r = 0; r < 3; r++) cpos[k][r] = mref[3 * r] * pc[0] + mref[3 * r + 1] * pc[1] + mref[3 * r + 2] * pc[2] + pref[r];
  }
  return naccept;
}

// one contact (slot) of bb_box_box_t: 1 when it exists (dist < 0), its distance, position and normal
template <class T>
__device__ __noinline__ int bb_slot_t(const T* p1, const T* R1, const float* s1, const T* p2, const T* R2,
                                      const float* s2, int slot, T* dist, T* pos, T* nrm) {
  T cd[BB_MAXVERT], cp[BB_MAXVERT][3];
  const int n = bb_box_box_t(p1, R1, s1, p2, R2, s2, 0.f, cd, cp, nrm);
  if (slot < 0 || slot >= n || !(cd[slot] < 0.f)) return 0;
  *dist = cd[slot];
  for (int k = 0; k < 3; k++) pos[k] = cp[slot][k];
  return 1;
}
"""
