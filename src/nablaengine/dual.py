"""Forward-mode duals for the engines' adjoints (device code): D1, a float value with one directional derivative, its
arithmetic (the value explicitly rounded: __fadd_rn, ...), the t_ math (t_fabs, t_sqrt, t_fmin, t_fmax; float and
double overloads too) and 3x3 helpers over any scalar (t_mv3, t_mtv3, t_mm3, t_mmt3, t_dot3, t_safe_div, t_rotmore_of).
A collision routine generated over D1 from its float text and replayed once per pose direction, with its branches fixed
by the values, gives the exact derivative of its outputs (forest_engine.model_hooks: box-box, meshes; g1_engine:
capsule-box terrain contacts)."""

DUAL_DEVICE = r"""
#ifndef DK_DUAL_DEVICE  // (included once: the forward's CV_ROUNDED mesh collision may bring it first)
#define DK_DUAL_DEVICE
struct D1 {
  float v, d;
  __device__ D1() {}
  __device__ D1(float x) : v(x), d(0.f) {}
  __device__ D1(float x, float y) : v(x), d(y) {}
};
__device__ __forceinline__ D1 operator+(D1 a, D1 b) { return D1(__fadd_rn(a.v, b.v), a.d + b.d); }
__device__ __forceinline__ D1 operator-(D1 a, D1 b) { return D1(__fsub_rn(a.v, b.v), a.d - b.d); }
__device__ __forceinline__ D1 operator*(D1 a, D1 b) { return D1(__fmul_rn(a.v, b.v), a.d * b.v + a.v * b.d); }
__device__ __forceinline__ D1 operator/(D1 a, D1 b) { return D1(__fdiv_rn(a.v, b.v), (a.d * b.v - a.v * b.d) / (b.v * b.v)); }
__device__ __forceinline__ D1 operator-(D1 a) { return D1(-a.v, -a.d); }
__device__ __forceinline__ D1 operator+(D1 a, float b) { return D1(__fadd_rn(a.v, b), a.d); }
__device__ __forceinline__ D1 operator+(float a, D1 b) { return D1(__fadd_rn(a, b.v), b.d); }
__device__ __forceinline__ D1 operator-(D1 a, float b) { return D1(__fsub_rn(a.v, b), a.d); }
__device__ __forceinline__ D1 operator-(float a, D1 b) { return D1(__fsub_rn(a, b.v), -b.d); }
__device__ __forceinline__ D1 operator*(D1 a, float b) { return D1(__fmul_rn(a.v, b), a.d * b); }
__device__ __forceinline__ D1 operator*(float a, D1 b) { return D1(__fmul_rn(a, b.v), a * b.d); }
__device__ __forceinline__ D1 operator/(D1 a, float b) { return D1(__fdiv_rn(a.v, b), a.d / b); }
__device__ __forceinline__ D1 operator/(float a, D1 b) { return D1(__fdiv_rn(a, b.v), -a * b.d / (b.v * b.v)); }
__device__ __forceinline__ D1& operator+=(D1& a, D1 b) { a = a + b; return a; }
__device__ __forceinline__ D1& operator-=(D1& a, D1 b) { a = a - b; return a; }
__device__ __forceinline__ D1& operator*=(D1& a, D1 b) { a = a * b; return a; }
__device__ __forceinline__ D1& operator/=(D1& a, D1 b) { a = a / b; return a; }
__device__ __forceinline__ bool operator<(D1 a, D1 b) { return a.v < b.v; }
__device__ __forceinline__ bool operator>(D1 a, D1 b) { return a.v > b.v; }
__device__ __forceinline__ bool operator<=(D1 a, D1 b) { return a.v <= b.v; }
__device__ __forceinline__ bool operator>=(D1 a, D1 b) { return a.v >= b.v; }
__device__ __forceinline__ bool operator<(D1 a, float b) { return a.v < b; }
__device__ __forceinline__ bool operator>(D1 a, float b) { return a.v > b; }
__device__ __forceinline__ bool operator<(float a, D1 b) { return a < b.v; }
__device__ __forceinline__ bool operator>(float a, D1 b) { return a > b.v; }
__device__ __forceinline__ bool operator!=(D1 a, float b) { return a.v != b; }
__device__ __forceinline__ bool operator==(D1 a, float b) { return a.v == b; }
__device__ __forceinline__ bool operator==(D1 a, D1 b) { return a.v == b.v; }
__device__ __forceinline__ bool operator!=(D1 a, D1 b) { return a.v != b.v; }
__device__ __forceinline__ bool operator<=(D1 a, float b) { return a.v <= b; }
__device__ __forceinline__ bool operator>=(D1 a, float b) { return a.v >= b; }
__device__ __forceinline__ bool operator<=(float a, D1 b) { return a <= b.v; }
__device__ __forceinline__ bool operator>=(float a, D1 b) { return a >= b.v; }
// the math the generated text calls (t_ names: the float builtins stay unshadowed)
__device__ __forceinline__ float t_fabs(float a) { return fabsf(a); }
__device__ __forceinline__ double t_fabs(double a) { return fabs(a); }
__device__ __forceinline__ double t_sqrt(double a) { return sqrt(a); }
__device__ __forceinline__ double t_fmin(double a, double b) { return fmin(a, b); }
__device__ __forceinline__ double t_fmax(double a, double b) { return fmax(a, b); }
__device__ __forceinline__ float t_sqrt(float a) { return sqrtf(a); }
__device__ __forceinline__ D1 t_fabs(D1 a) { return a.v < 0.f ? -a : a; }
__device__ __forceinline__ D1 t_sqrt(D1 a) { float r = __fsqrt_rn(a.v); return D1(r, r > 0.f ? 0.5f * a.d / r : 0.f); }
__device__ __forceinline__ float t_fmin(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ float t_fmax(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ D1 t_fmin(D1 a, D1 b) { return b.v < a.v ? b : a; }
__device__ __forceinline__ D1 t_fmax(D1 a, D1 b) { return b.v > a.v ? b : a; }
__device__ __forceinline__ D1 t_fmin(D1 a, float b) { return b < a.v ? D1(b) : a; }
__device__ __forceinline__ D1 t_fmax(D1 a, float b) { return b > a.v ? D1(b) : a; }
__device__ __forceinline__ D1 t_fmin(float a, D1 b) { return t_fmin(b, a); }
__device__ __forceinline__ D1 t_fmax(float a, D1 b) { return t_fmax(b, a); }
template <class A, class V, class O>
__device__ __forceinline__ void t_mv3(const A* M, const V* v, O* o) {
  for (int i = 0; i < 3; i++) o[i] = M[3 * i] * v[0] + M[3 * i + 1] * v[1] + M[3 * i + 2] * v[2];
}
template <class A, class V, class O>
__device__ __forceinline__ void t_mtv3(const A* M, const V* v, O* o) {
  for (int i = 0; i < 3; i++) o[i] = M[i] * v[0] + M[3 + i] * v[1] + M[6 + i] * v[2];
}
template <class A, class B, class C>
__device__ __forceinline__ void t_mm3(const A* M, const B* N, C* o) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) o[3 * i + j] = M[3 * i] * N[j] + M[3 * i + 1] * N[3 + j] + M[3 * i + 2] * N[6 + j];
}
template <class A, class B, class C>
__device__ __forceinline__ void t_mmt3(const A* M, const B* N, C* o) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      o[3 * i + j] = M[3 * i] * N[3 * j] + M[3 * i + 1] * N[3 * j + 1] + M[3 * i + 2] * N[3 * j + 2];
}
template <class A, class B>
__device__ __forceinline__ A t_dot3(const A* a, const B* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
template <class X, class T>
__device__ __forceinline__ T t_safe_div(X x, T y) { return x / (y != 0.f ? y : T(1e-15f)); }
template <class T>
__device__ __forceinline__ void t_rotmore_of(int f, T* r) {
  float rf[9];
  rotmore_of(f, rf);
  for (int k = 0; k < 9; k++) r[k] = T(rf[k]);
}
#endif  // DK_DUAL_DEVICE
"""

RF_DEVICE = r"""
struct RF {
  float v;
  __device__ RF() {}
  __device__ RF(float x) : v(x) {}
};
__device__ __forceinline__ RF operator+(RF a, RF b) { return RF(__fadd_rn(a.v, b.v)); }
__device__ __forceinline__ RF operator-(RF a, RF b) { return RF(__fsub_rn(a.v, b.v)); }
__device__ __forceinline__ RF operator*(RF a, RF b) { return RF(__fmul_rn(a.v, b.v)); }
__device__ __forceinline__ RF operator/(RF a, RF b) { return RF(__fdiv_rn(a.v, b.v)); }
__device__ __forceinline__ RF operator-(RF a) { return RF(-a.v); }
__device__ __forceinline__ RF operator+(RF a, float b) { return RF(__fadd_rn(a.v, b)); }
__device__ __forceinline__ RF operator+(float a, RF b) { return RF(__fadd_rn(a, b.v)); }
__device__ __forceinline__ RF operator-(RF a, float b) { return RF(__fsub_rn(a.v, b)); }
__device__ __forceinline__ RF operator-(float a, RF b) { return RF(__fsub_rn(a, b.v)); }
__device__ __forceinline__ RF operator*(RF a, float b) { return RF(__fmul_rn(a.v, b)); }
__device__ __forceinline__ RF operator*(float a, RF b) { return RF(__fmul_rn(a, b.v)); }
__device__ __forceinline__ RF operator/(RF a, float b) { return RF(__fdiv_rn(a.v, b)); }
__device__ __forceinline__ RF operator/(float a, RF b) { return RF(__fdiv_rn(a, b.v)); }
__device__ __forceinline__ RF& operator+=(RF& a, RF b) { a = a + b; return a; }
__device__ __forceinline__ RF& operator-=(RF& a, RF b) { a = a - b; return a; }
__device__ __forceinline__ RF& operator*=(RF& a, RF b) { a = a * b; return a; }
__device__ __forceinline__ RF& operator/=(RF& a, RF b) { a = a / b; return a; }
__device__ __forceinline__ bool operator<(RF a, RF b) { return a.v < b.v; }
__device__ __forceinline__ bool operator>(RF a, RF b) { return a.v > b.v; }
__device__ __forceinline__ bool operator<=(RF a, RF b) { return a.v <= b.v; }
__device__ __forceinline__ bool operator>=(RF a, RF b) { return a.v >= b.v; }
__device__ __forceinline__ bool operator<(RF a, float b) { return a.v < b; }
__device__ __forceinline__ bool operator>(RF a, float b) { return a.v > b; }
__device__ __forceinline__ bool operator<(float a, RF b) { return a < b.v; }
__device__ __forceinline__ bool operator>(float a, RF b) { return a > b.v; }
__device__ __forceinline__ bool operator!=(RF a, float b) { return a.v != b; }
__device__ __forceinline__ bool operator==(RF a, float b) { return a.v == b; }
__device__ __forceinline__ bool operator==(RF a, RF b) { return a.v == b.v; }
__device__ __forceinline__ bool operator!=(RF a, RF b) { return a.v != b.v; }
__device__ __forceinline__ bool operator<=(RF a, float b) { return a.v <= b; }
__device__ __forceinline__ bool operator>=(RF a, float b) { return a.v >= b; }
__device__ __forceinline__ bool operator<=(float a, RF b) { return a <= b.v; }
__device__ __forceinline__ bool operator>=(float a, RF b) { return a >= b.v; }
__device__ __forceinline__ RF t_fabs(RF a) { return a.v < 0.f ? -a : a; }
__device__ __forceinline__ RF t_sqrt(RF a) { return RF(__fsqrt_rn(a.v)); }
__device__ __forceinline__ RF t_fmin(RF a, RF b) { return b.v < a.v ? b : a; }
__device__ __forceinline__ RF t_fmax(RF a, RF b) { return b.v > a.v ? b : a; }
__device__ __forceinline__ RF t_fmin(RF a, float b) { return b < a.v ? RF(b) : a; }
__device__ __forceinline__ RF t_fmax(RF a, float b) { return b > a.v ? RF(b) : a; }
__device__ __forceinline__ RF t_fmin(float a, RF b) { return t_fmin(b, a); }
__device__ __forceinline__ RF t_fmax(float a, RF b) { return t_fmax(b, a); }
"""



def dn_device(ks=(1, 2, 3, 4, 6, 12)):
    """DN: a float value with DN_K directional derivatives (DN_K, a compile-time constant set by the includer, one of
    ks), the tangents scalar members d0, d1, ... (explicit per-tangent statements, no loops or arrays), each by D1's
    expression, the value explicitly rounded once: a routine generated over DN computes in one pass the values and the
    DN_K tangents of DN_K D1 runs, with the branches fixed by the values. The operators are D1's text (constructor
    form), but the tangents are D1's bits only where nvcc contracts them into FMAs alike, which depends on the code
    around them (g1_engine.cylccd.NT_HEAD). DN_D(x, k): tangent k (k a runtime index)."""
    def each(k, f):
        return " ".join(f(f"d{i}") for i in range(k))
    def binop(op, vop, tan):
        return lambda k: (f"__device__ __forceinline__ DN operator{op} {{ return DN({vop}, "
                          + ", ".join(tan(f"d{i}") for i in range(k)) + "); }\n")
    ops = [
        binop("+(DN a, DN b)", "__fadd_rn(a.v, b.v)", lambda d: f"a.{d} + b.{d}"),
        binop("-(DN a, DN b)", "__fsub_rn(a.v, b.v)", lambda d: f"a.{d} - b.{d}"),
        binop("*(DN a, DN b)", "__fmul_rn(a.v, b.v)", lambda d: f"a.{d} * b.v + a.v * b.{d}"),
        binop("/(DN a, DN b)", "__fdiv_rn(a.v, b.v)", lambda d: f"(a.{d} * b.v - a.v * b.{d}) / (b.v * b.v)"),
        binop("-(DN a)", "-a.v", lambda d: f"-a.{d}"),
        binop("+(DN a, float b)", "__fadd_rn(a.v, b)", lambda d: f"a.{d}"),
        binop("+(float a, DN b)", "__fadd_rn(a, b.v)", lambda d: f"b.{d}"),
        binop("-(DN a, float b)", "__fsub_rn(a.v, b)", lambda d: f"a.{d}"),
        binop("-(float a, DN b)", "__fsub_rn(a, b.v)", lambda d: f"-b.{d}"),
        binop("*(DN a, float b)", "__fmul_rn(a.v, b)", lambda d: f"a.{d} * b"),
        binop("*(float a, DN b)", "__fmul_rn(a, b.v)", lambda d: f"a * b.{d}"),
        binop("/(DN a, float b)", "__fdiv_rn(a.v, b)", lambda d: f"a.{d} / b"),
        binop("/(float a, DN b)", "__fdiv_rn(a, b.v)", lambda d: f"-a * b.{d} / (b.v * b.v)"),
    ]
    out = ["#ifndef DK_DN_DEVICE\n#define DK_DN_DEVICE\n"]
    for k in ks:
        out.append(f"#{'if' if k == ks[0] else 'elif'} DN_K == {k}\n")
        out.append("struct DN {\n  float v, " + ", ".join(f"d{i}" for i in range(k)) + ";\n  __device__ DN() {}\n"
                   "  __device__ DN(float x) : v(x), " + ", ".join(f"d{i}(0.f)" for i in range(k)) + " {}\n"
                   "  __device__ DN(float x, " + ", ".join(f"float y{i}" for i in range(k)) + ") : v(x), "
                   + ", ".join(f"d{i}(y{i})" for i in range(k)) + " {}\n};\n")
        out.append("__device__ __forceinline__ float& DN_D(DN& x, int k) {\n  switch (k) {\n"
                   + "".join(f"    case {i}: return x.d{i};\n" for i in range(1, k)) + "    default: return x.d0;\n  }\n}\n")
        out.append("__device__ __forceinline__ float DN_D(const DN& x, int k) {\n  switch (k) {\n"
                   + "".join(f"    case {i}: return x.d{i};\n" for i in range(1, k)) + "    default: return x.d0;\n  }\n}\n")
        out += [f(k) for f in ops]
        out.append("__device__ __forceinline__ DN t_sqrt(DN a) { float r = __fsqrt_rn(a.v); return DN(r, "
                   + ", ".join(f"r > 0.f ? 0.5f * a.d{i} / r : 0.f" for i in range(k)) + "); }\n")
        out.append("#define DN_SIGNED_TANGENTS(o, s, x) do { "
                   + each(k, lambda d: f"(o).{d} = (s) * (x).{d};") + " } while (0)\n")
    out.append(f"#else\n#error \"DN_K: one of {', '.join(map(str, ks))}\"\n#endif\n")
    out.append(r"""__device__ __forceinline__ DN& operator+=(DN& a, DN b) { a = a + b; return a; }
__device__ __forceinline__ DN& operator-=(DN& a, DN b) { a = a - b; return a; }
__device__ __forceinline__ DN& operator*=(DN& a, DN b) { a = a * b; return a; }
__device__ __forceinline__ DN& operator/=(DN& a, DN b) { a = a / b; return a; }
__device__ __forceinline__ bool operator<(DN a, DN b) { return a.v < b.v; }
__device__ __forceinline__ bool operator>(DN a, DN b) { return a.v > b.v; }
__device__ __forceinline__ bool operator<=(DN a, DN b) { return a.v <= b.v; }
__device__ __forceinline__ bool operator>=(DN a, DN b) { return a.v >= b.v; }
__device__ __forceinline__ bool operator<(DN a, float b) { return a.v < b; }
__device__ __forceinline__ bool operator>(DN a, float b) { return a.v > b; }
__device__ __forceinline__ bool operator<(float a, DN b) { return a < b.v; }
__device__ __forceinline__ bool operator>(float a, DN b) { return a > b.v; }
__device__ __forceinline__ bool operator!=(DN a, float b) { return a.v != b; }
__device__ __forceinline__ bool operator==(DN a, float b) { return a.v == b; }
__device__ __forceinline__ bool operator==(DN a, DN b) { return a.v == b.v; }
__device__ __forceinline__ bool operator!=(DN a, DN b) { return a.v != b.v; }
__device__ __forceinline__ bool operator<=(DN a, float b) { return a.v <= b; }
__device__ __forceinline__ bool operator>=(DN a, float b) { return a.v >= b; }
__device__ __forceinline__ bool operator<=(float a, DN b) { return a <= b.v; }
__device__ __forceinline__ bool operator>=(float a, DN b) { return a >= b.v; }
__device__ __forceinline__ DN t_fabs(DN a) { return a.v < 0.f ? -a : a; }
__device__ __forceinline__ DN t_fmin(DN a, DN b) { return b.v < a.v ? b : a; }
__device__ __forceinline__ DN t_fmax(DN a, DN b) { return b.v > a.v ? b : a; }
__device__ __forceinline__ DN t_fmin(DN a, float b) { return b < a.v ? DN(b) : a; }
__device__ __forceinline__ DN t_fmax(DN a, float b) { return b > a.v ? DN(b) : a; }
__device__ __forceinline__ DN t_fmin(float a, DN b) { return t_fmin(b, a); }
__device__ __forceinline__ DN t_fmax(float a, DN b) { return t_fmax(b, a); }
#endif  // DK_DN_DEVICE
""")
    return "".join(out)
