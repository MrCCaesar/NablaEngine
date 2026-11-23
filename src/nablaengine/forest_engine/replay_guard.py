"""The contact adjoint's guard against a contact its replay does not reproduce (the forest engine)."""
import re

from nablaengine.g1_engine.replay_guard import CALL_NEW, CALL_RE, ONE_HEAD, guard_one, primitive_only

CALL_SUR = ("      bad = (zone || dbg) && !(d_ < 0.f);  // (the replay check: contacts with a quadratic row, as without it)\n"
            "      if (zone || s.sur_mu > 0.f) cb = c_gbody[PG1(pair)] | c_gbody[PG2(pair)] << 8;\n",
            "      bad = ((zone || dbg) && !(d_ < 0.f)) || d_ == ADJ_NO_REPLAY;  // (the replay check: contacts with a "
            "quadratic row,\n      // as without it, and any contact the replay does not find: replay_guard)\n"
            "      if ((zone || s.sur_mu > 0.f) && d_ != ADJ_NO_REPLAY) cb = c_gbody[PG1(pair)] | c_gbody[PG2(pair)] << 8;\n")
SUR_RE = re.compile(r"\n( *)(adj_pair_geom\(pair, slot, s\.sgx, s\.sgq, &d, p, fr, &nu, nullptr, nullptr[^\n]*\));\n")


def needed(m):
    """A contact pair with a geom other than plane, sphere and capsule."""
    return not primitive_only(m)


def apply(src, m, terrain=False):
    """src (adjoint.cuda_source's text) with the guard where needed(m) or in a terrain build (terrain: the terrain's
    boxes and heightfields, forest_engine.terrain); else unchanged. A text without the adjoint (no adj_contact_one) is
    returned unchanged."""
    if not (needed(m) or terrain) or ONE_HEAD not in src:
        return src
    src = guard_one(src)
    n_plain, n_sur = len(CALL_RE.findall(src)), src.count(CALL_SUR[0])
    if (n_plain, n_sur) not in ((1, 0), (0, 1)):
        raise RuntimeError(f"replay_guard: adj_contacts' call found {n_plain} (plain) / {n_sur} (surrogate) times")
    if n_plain:
        src = CALL_RE.sub(lambda mt: CALL_NEW.format(args=mt.group(1)), src)
    else:
        src = src.replace(*CALL_SUR)
    if n_sur:
        hits = SUR_RE.findall(src)
        if len(hits) != 1:
            raise RuntimeError(f"replay_guard: sur_contact_coef's replay call found {len(hits)} times")
        src = SUR_RE.sub(lambda mt: (f"\n{mt.group(1)}if ({mt.group(2)} < 0) {{  // (replay_guard: no contact `slot`)\n"
                                     f"{mt.group(1)}  for (int t = 0; t < 18; t++) cw[t] = 0.f;\n"
                                     f"{mt.group(1)}  return;\n{mt.group(1)}}}\n"), src)
    return src
