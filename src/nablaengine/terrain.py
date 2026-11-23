"""Terrain for the engine: static boxes and heightfields shared by every world."""
from dataclasses import dataclass, field

import mujoco
import numpy as np

BOX_WORDS = 15
HF_WORDS = 16
HF_INTS = 4
OBJ_WORDS = 15
OBJ_VEL_WORDS = 18
OBJ_MAT_WORDS = 7
TER_MAT_WORDS = 8
TER_MAT_MODES = {"none": 0, "max": 1, "override": 2}
TER_SAND_WORDS = 4
TER_MAT_SAND_WORDS = TER_MAT_WORDS + TER_SAND_WORDS
OBJ_E0 = 0xFF00
MAX_OBJECTS = 64


def pack_com(com, objects, velocities=None, materials=None):
    """com (B, 3) and objects (B, K, OBJ_WORDS) -> the com input of a build with K object slots, (B, 3 + OBJ_WORDS K)
    (NumPy, JAX or torch arrays; the concatenation of the library's namespace). velocities (B, K, OBJ_VEL_WORDS), a
    build with TerrainConfig.object_velocity: each slot's motion words (TerrainConfig; slot_velocity), appended after
    all slots: (B, 3 + (OBJ_WORDS + OBJ_VEL_WORDS) K). materials (B, K, OBJ_MAT_WORDS), a build with
    TerrainConfig.object_material: each slot's contact material (slot_material), after the velocities."""
    B = com.shape[0]
    parts = ([com, objects.reshape(B, -1)] + ([] if velocities is None else [velocities.reshape(B, -1)])
             + ([] if materials is None else [materials.reshape(B, -1)]))
    if hasattr(com, "new_zeros"):
        import torch
        return torch.cat(parts, 1).contiguous()
    try:
        import jax.numpy as jnp
        if not isinstance(com, np.ndarray):
            return jnp.concatenate(parts, 1)
    except ImportError:
        pass
    return np.concatenate(parts, 1)


def slot_velocity(v=(0.0, 0.0, 0.0), w=(0.0, 0.0, 0.0), s=(0.0, 0.0, 0.0), a=(0.0, 0.0, 0.0), al=(0.0, 0.0, 0.0),
                  sd=(0.0, 0.0, 0.0)):
    """One slot's motion words (OBJ_VEL_WORDS,) for pack_com (TerrainConfig.object_velocity; NumPy), at the start of
    the control step: v, w the slot's linear velocity at its position and angular velocity (world frame); s a surface
    velocity in the slot's frame (its tangential part at each contact; the pose stays); a, al the linear and angular
    acceleration (world frame); sd the rate of s. Within the control step the engine moves the slot and its velocities
    with them (MuJoCo's semi-implicit Euler: v_k = v + k h a, pos_k = pos + h sum_{j<=k} v_j), and the contacts take
    the slot's velocity and acceleration (a moving body's J v and J qacc in MuJoCo's contact rows)."""
    return np.concatenate([np.asarray(x, np.float32) for x in (v, w, s, a, al, sd)])


def slot_material(friction=1.0, solref=(0.02, 1.0), solimp=(0.9, 0.95, 0.001, 0.5)):
    """One slot's contact material (OBJ_MAT_WORDS,) for pack_com (TerrainConfig.object_material; NumPy): the sliding
    friction mu, solref (time constant, damping ratio; or MuJoCo's direct form, each entry <= 0: -stiffness,
    -damping) and solimp (dmin, dmax, width, midpoint; the power stays the terrain pairs' own) of every contact of the
    slot, in place of the robot geom's terrain-pair parameters (and of the friction input / foot_solref / foot_solimp
    for foot pairs). The defaults are MuJoCo's (geom friction 1, solref (0.02, 1), solimp (0.9, 0.95, 0.001, 0.5))."""
    sr, si = np.asarray(solref, np.float32).reshape(2), np.asarray(solimp, np.float32).reshape(-1)
    if si.size == 5:
        si = si[:4]
    if si.size != 4:
        raise ValueError("slot_material: solimp (dmin, dmax, width, midpoint)")
    return np.concatenate([np.asarray([friction], np.float32), sr, si])


def terrain_material(friction=1.0, mode="max", solref=(0.02, 1.0), solimp=(0.9, 0.95, 0.001, 0.5), sand=None):
    """One terrain element's material (TER_MAT_WORDS,) for Terrain.material (TerrainConfig.terrain_material; NumPy);
    with sand (sand_fields(), or its arguments as a tuple (k, s_max, yield_force, rate)) the row of TER_MAT_SAND_WORDS
    with the sand fields after the material (TerrainConfig.terrain_sand).
    mode "max": the element's contacts take max(the robot side's friction, friction), as MuJoCo mixes two geoms'
    frictions (the robot side: the geom's floor pair's mu, or for a foot pair the per-world friction input); solref /
    solimp stay the pair's. mode "override": the contacts take friction, solref and solimp outright (the friction input
    and foot_solref / foot_solimp included), as MuJoCo does for a geom of higher priority. mode "none": the pair's
    parameters (the default of every element)."""
    if mode not in TER_MAT_MODES:
        raise ValueError(f"terrain_material: mode {mode!r}: one of {tuple(TER_MAT_MODES)}")
    sr, si = np.asarray(solref, np.float32).reshape(2), np.asarray(solimp, np.float32).reshape(-1)[:4]
    row = np.concatenate([np.asarray([friction], np.float32), sr, si, np.asarray([TER_MAT_MODES[mode]], np.float32)])
    if sand is None:
        return row
    sd = np.asarray(sand, np.float32).reshape(-1)
    if sd.size != TER_SAND_WORDS:
        raise ValueError("terrain_material: sand (k, s_max, yield_force, rate)")
    return np.concatenate([row, sand_fields(*sd)])


def sand_fields(k=1.0, s_max=0.0, yield_force=0.0, rate=0.0):
    """One element's sand fields (TER_SAND_WORDS,): k in [0, 1] (0 hard ground, 1 full sand: TerrainConfig.terrain_sand),
    s_max (m), the yield force F_y (body weights per foot) and the sinkage rate (m/s per body weight of normal force
    above the yield). The engine applies k (the foot_offset sinkage k s(x)); it returns s_max, F_y and the rate per foot
    (contact_table's foot_sand_params) without using them."""
    v = np.asarray([k, s_max, yield_force, rate], np.float64)
    if not 0.0 <= v[0] <= 1.0 or (v[1:] < 0).any() or not np.isfinite(v).all():
        raise ValueError(f"sand_fields: k in [0, 1], s_max / yield_force / rate >= 0 (got {v.tolist()})")
    return v.astype(np.float32)


def box_object(pos, rot, half):
    """One object slot (OBJ_WORDS,) from a position, a rotation matrix (3, 3) and half sizes (NumPy)."""
    return np.concatenate([np.asarray(pos, np.float32), np.asarray(rot, np.float32).reshape(9),
                           np.asarray(half, np.float32)])


@dataclass
class TerrainConfig:
    """Compile-time terrain options. geoms: "feet" (the robot geoms of the foot-floor pairs), "body" (every robot
    geom with a floor pair) or a list of robot geoms (names or geom ids; each needs a floor pair, whose parameters its
    terrain contacts take). The terrain broad phase handles one geom per lane of a warp: up to 32 geoms take one pass,
    more a second (the G1's "body" has 33: a list of at most 32 avoids the second pass); floor: keep the model's floor plane pairs too (a terrain usually has its own ground);
    cap_points: heightfield samples per capsule; mode: heightfield contacts, "analytic"
    (sample points against the triangle below) or "native" (MuJoCo's prisms, each measured
    exactly, at most 4 contacts per geom and heightfield: step.py, TERRAIN_HF_NATIVE); maxcon: contact capacity per
    world (None: the model's, step.maxcon); objects: per-world box slots (OBJ_WORDS floats each: pos, rot row major, half
    sizes; half size 0 = absent) given with each step after the COM offset, com (B, 3 + OBJ_WORDS objects): kinematic
    obstacles / targets whose poses the caller sets per world at any step (no recompilation), colliding with the same
    robot geoms as the terrain (pack_com); object_velocity (forest engine): each slot also has motion words
    (OBJ_VEL_WORDS, pack_com's velocities, slot_velocity): v, w the slot's rigid velocity (world frame, v at its
    position), s a surface velocity in the slot's frame, a, al the rigid motion's accelerations, sd the rate of s.
    Within a control step the engine moves the slot with them (semi-implicit Euler per substep; the caller sets the
    next step's pose), and at a contact point p its surface moves with u = v + w x (p - pos) + the tangential part of
    R s and accelerates with a + al x (p - pos) + the tangential part of R sd: the contact rows act on the robot's
    motion relative to the slot's (a moving platform, a pusher: (v, w, a, al); a conveyor belt or an escalator: a
    fixed pose and s, sd). Off: the slots are at rest (the default; the build and com layout unchanged).
    terrain_material (forest engine): the static terrain's elements may carry a material (Terrain.material,
    terrain_material: friction mixed as MuJoCo's max, or friction / solref / solimp overriding, as a geom of higher
    priority); not differentiable (the terrain is static). Off: every element takes the pair's parameters (the default;
    the build unchanged).
    object_material (forest engine): each slot also has its contact material (OBJ_MAT_WORDS, pack_com's materials,
    slot_material: friction mu, solref, solimp) per world, used by every contact of the slot in place of the terrain
    pair's; the step's gradients include the words' cotangents. Off: the slots' contacts take the terrain pairs'
    parameters (the default; the build and com layout unchanged).
    terrain_sand (forest engine, with the foot_offset parameter): the elements' material rows
    carry sand fields (TER_MAT_SAND_WORDS: terrain_material(sand=...), sand_fields: k, s_max, F_y, rate; 8-word rows
    and elements without a row are hard ground, k = 0): a foot contact on element e takes the distance d + k_e s(x),
    s(x) the per-world foot_offset; contact_table splits foot_normal into the sand and hard parts and returns the feet's
    sand presence and sand parameters (docs/terrain.md). sand_other: (k, s_max, F_y, rate) of the foot contacts with
    anything other than a terrain element (the floor plane, the object slots): k = 1 (the default) keeps foot_offset's
    behaviour there. The material words (friction, solref, solimp, mode) act only with terrain_material as well (without
    it, as there, every mode must be "none"). k is piecewise per element (no blend at element edges; a ledge in the
    support of k s at a sand / hard boundary, physical). Off: the build unchanged."""
    geoms: object = "feet"
    floor: bool = False
    cap_points: int = 3
    mode: str = "analytic"
    maxcon: int = None
    objects: int = 0
    object_velocity: bool = False
    object_material: bool = False
    terrain_material: bool = False
    terrain_sand: bool = False
    sand_other: tuple = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        if isinstance(self.geoms, str):
            if self.geoms not in ("feet", "body"):
                raise ValueError(f"TerrainConfig.geoms {self.geoms!r}: 'feet', 'body' or a list of geom names / ids")
        else:
            try:
                gs = list(self.geoms)
            except TypeError:
                raise ValueError(f"TerrainConfig.geoms {self.geoms!r}: 'feet', 'body' or a list of geom names / ids")
            if not gs or not all(isinstance(g, str) or (isinstance(g, (int, np.integer)) and not isinstance(g, bool))
                                 for g in gs):
                raise ValueError(f"TerrainConfig.geoms {self.geoms!r}: a non-empty list of geom names (str) or ids (int)")
            self.geoms = tuple(g if isinstance(g, str) else int(g) for g in gs)
        if not 0 <= int(self.objects) <= MAX_OBJECTS:
            raise ValueError(f"TerrainConfig.objects {self.objects}: 0 .. {MAX_OBJECTS}")
        if self.object_velocity and not int(self.objects):
            raise ValueError("TerrainConfig.object_velocity: needs object slots (objects > 0)")
        if self.object_material and not int(self.objects):
            raise ValueError("TerrainConfig.object_material: needs object slots (objects > 0)")
        if self.mode not in ("analytic", "native"):
            raise ValueError(f"TerrainConfig.mode {self.mode!r}: 'analytic' (sample points) or 'native' (prisms)")
        self.sand_other = tuple(float(x) for x in sand_fields(*self.sand_other))

    def key(self):
        k = dict(geoms=self.geoms if isinstance(self.geoms, str) else list(self.geoms), floor=self.floor, cap_points=self.cap_points, mode=self.mode,
                 maxcon=self.maxcon)
        if self.objects:
            k["objects"] = int(self.objects)
        if self.object_velocity:
            k["object_velocity"] = True
        if self.object_material:
            k["object_material"] = True
        if self.terrain_material:
            k["terrain_material"] = True
        if self.terrain_sand:
            k["terrain_sand"] = True
            k["sand_other"] = list(self.sand_other)
        return k

    def velocity_words(self):
        """Floats per world of the slots' velocities (object_velocity: OBJ_VEL_WORDS per slot), else 0."""
        return OBJ_VEL_WORDS * int(self.objects) if self.object_velocity else 0

    def material_words(self):
        """Floats per world of the slots' contact materials (object_material: OBJ_MAT_WORDS per slot), else 0."""
        return OBJ_MAT_WORDS * int(self.objects) if self.object_material else 0

    def com_words(self):
        """Floats per world of the com input: the COM offset (3), the object slots, their velocities and materials."""
        return 3 + (OBJ_WORDS + (OBJ_VEL_WORDS if self.object_velocity else 0)) * int(self.objects) + self.material_words()


@dataclass
class Terrain:
    """Static terrain data in world coordinates (float64 here, float32 on the device)."""
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, BOX_WORDS)))
    hf_head: np.ndarray = field(default_factory=lambda: np.zeros((0, HF_WORDS)))
    hf_int: np.ndarray = field(default_factory=lambda: np.zeros((0, HF_INTS), np.int64))
    hf_data: np.ndarray = field(default_factory=lambda: np.zeros(0))
    material: np.ndarray = None

    @staticmethod
    def from_model(m, d=None):
        """The static box and hfield geoms of MjModel m: those of the world body and of bodies welded to it (mjlab's
        "terrain" body; poses from mj_kinematics on d, or a fresh MjData)."""
        if d is None:
            d = mujoco.MjData(m)
            mujoco.mj_kinematics(m, d)
        boxes, head, ints, data = [], [], [], []
        adr = 0
        for g in range(m.ngeom):
            if m.body_weldid[m.geom_bodyid[g]] != 0:
                continue
            t = int(m.geom_type[g])
            pos, rot = d.geom_xpos[g].copy(), d.geom_xmat[g].copy()
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                boxes.append(np.concatenate([pos, rot, m.geom_size[g]]))
            elif t == mujoco.mjtGeom.mjGEOM_HFIELD:
                h = int(m.geom_dataid[g])
                nrow, ncol = int(m.hfield_nrow[h]), int(m.hfield_ncol[h])
                a = int(m.hfield_adr[h])
                head.append(np.concatenate([pos, rot, m.hfield_size[h]]))
                ints.append([nrow, ncol, adr, 0])
                data.append(m.hfield_data[a:a + nrow * ncol])
                adr += nrow * ncol
        t = Terrain()
        if boxes:
            t.boxes = np.asarray(boxes, np.float64)
        if head:
            t.hf_head, t.hf_int = np.asarray(head, np.float64), np.asarray(ints, np.int64)
            t.hf_data = np.concatenate(data).astype(np.float64)
        return t

    def save(self, path):
        """Write the terrain to an .npz file (load() reads it back)."""
        np.savez(path, boxes=self.boxes, hf_head=self.hf_head, hf_int=self.hf_int, hf_data=self.hf_data,
                 **({"material": self.material} if self.material is not None else {}))

    @staticmethod
    def load(path):
        z = np.load(path)
        return Terrain(boxes=z["boxes"], hf_head=z["hf_head"], hf_int=z["hf_int"], hf_data=z["hf_data"],
                       material=z["material"] if "material" in z.files else None)

    def add_to_spec(self, spec, prefix="ter"):
        """Add the terrain to an MjSpec's world body as box and hfield geoms (the reverse of from_model, e.g. for a
        MuJoCo / MuJoCo Warp reference scene); returns the geom names."""
        import mujoco
        names = []
        wb = spec.worldbody
        for i, b in enumerate(self.boxes):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(b[3:12], np.float64))
            n = f"{prefix}_box{i}"
            wb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_BOX, size=b[12:15].tolist(), pos=b[:3].tolist(), quat=q.tolist())
            names.append(n)
        for j, (h, hi) in enumerate(zip(self.hf_head, self.hf_int)):
            nrow, ncol, adr = int(hi[0]), int(hi[1]), int(hi[2])
            n = f"{prefix}_hf{j}"
            spec.add_hfield(name=n, size=h[12:16].tolist(), nrow=nrow, ncol=ncol,
                            userdata=self.hf_data[adr:adr + nrow * ncol].tolist())
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(h[3:12], np.float64))
            wb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=n, pos=h[:3].tolist(), quat=q.tolist())
            names.append(n)
        return names

    def _material_rows(self):
        n = len(self.boxes) + len(self.hf_head)
        if self.material is None:
            return None
        a = np.asarray(self.material, np.float32)
        if a.ndim != 2 or a.shape[1] not in (TER_MAT_WORDS, TER_MAT_SAND_WORDS):
            raise ValueError(f"Terrain.material: rows of {TER_MAT_WORDS} words (terrain_material) or "
                             f"{TER_MAT_SAND_WORDS} (with the sand fields), got shape {a.shape}")
        if len(a) != n:
            raise ValueError(f"Terrain.material: {len(a)} rows for {n} elements (the boxes, then the heightfields)")
        if a.shape[1] == TER_MAT_WORDS:
            a = np.concatenate([a, np.zeros((n, TER_SAND_WORDS), np.float32)], 1)
        return a

    def material_array(self):
        """(n elements, TER_MAT_WORDS) float32: Terrain.material's material words, mode 0 (none) where it is None."""
        n = len(self.boxes) + len(self.hf_head)
        a = self._material_rows()
        if a is None:
            return np.zeros((n, TER_MAT_WORDS), np.float32)
        a = np.ascontiguousarray(a[:, :TER_MAT_WORDS])
        if not np.isin(a[:, 7], (0, 1, 2)).all():
            raise ValueError("Terrain.material: mode words 0, 1 or 2 (terrain_material)")
        return a

    def sand_array(self):
        """(n elements, TER_SAND_WORDS) float32: the elements' sand fields (k, s_max, F_y, rate); 0 (hard ground) for
        8-word rows or material None."""
        n = len(self.boxes) + len(self.hf_head)
        a = self._material_rows()
        if a is None:
            return np.zeros((n, TER_SAND_WORDS), np.float32)
        sd = np.ascontiguousarray(a[:, TER_MAT_WORDS:])
        if not (np.isfinite(sd).all() and (sd[:, 0] >= 0).all() and (sd[:, 0] <= 1).all() and (sd[:, 1:] >= 0).all()):
            raise ValueError("Terrain.material: sand fields k in [0, 1], s_max / yield / rate >= 0 (sand_fields)")
        return sd

    def material_upload(self, cfg=None):
        """The device's element words for a build's TerrainConfig (or its key() dict): (n, 8) material words; with
        terrain_sand (n, 12) the material and sand fields."""
        get = (cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))) if cfg is not None else \
            (lambda k, d=None: d)
        mat = self.material_array()
        if not get("terrain_sand", False):
            return mat
        return np.ascontiguousarray(np.concatenate([mat, self.sand_array()], 1).astype(np.float32))

    def element_aabbs(self):
        """(n, 4) xmin, ymin, xmax, ymax of the elements: the boxes, then the heightfields."""
        out = []
        for b in self.boxes:
            R, hs = b[3:12].reshape(3, 3), b[12:15]
            ext = np.abs(R) @ hs
            out.append([b[0] - ext[0], b[1] - ext[1], b[0] + ext[0], b[1] + ext[1]])
        for h in self.hf_head:
            R, sz = h[3:12].reshape(3, 3), h[12:16]
            ext = np.abs(R[:2, :2]) @ sz[:2]
            out.append([h[0] - ext[0], h[1] - ext[1], h[0] + ext[0], h[1] + ext[1]])
        return np.asarray(out, np.float64).reshape(-1, 4)

    def pack(self, cell=1.0):
        """Device arrays: boxes float32 (nbox, 15), heightfield headers float32 (nhf, 16) and ints int32 (nhf, 4), data
        float32, and the broad-phase grid: grid float32 [x0, y0, 1 / cell, nx, ny, nbox], cell_start int32 (nx ny + 1),
        cell_items int32 (element index: box i, or nbox + heightfield j), elem_cell int32 (n, 2): the element's first
        cell (cx, cy)."""
        aabb = self.element_aabbs()
        nbox = len(self.boxes)
        if len(aabb) >= OBJ_E0:
            raise ValueError(f"{len(aabb)} terrain elements: at most {OBJ_E0 - 1} (the tape's 16-bit element index, "
                             "the per-world objects above)")
        if len(aabb):
            x0, y0 = aabb[:, 0].min() - 1e-3, aabb[:, 1].min() - 1e-3
            nx = max(1, int(np.ceil((aabb[:, 2].max() - x0) / cell)) + 1)
            ny = max(1, int(np.ceil((aabb[:, 3].max() - y0) / cell)) + 1)
        else:
            x0 = y0 = 0.0
            nx = ny = 1
        lists = [[] for _ in range(nx * ny)]
        first = np.zeros((len(aabb), 2), np.int32)
        for e, (a0, b0, a1, b1) in enumerate(aabb):
            c0, c1 = int((a0 - x0) / cell), min(nx - 1, int((a1 - x0) / cell))
            r0, r1 = int((b0 - y0) / cell), min(ny - 1, int((b1 - y0) / cell))
            first[e] = (c0, r0)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    lists[r * nx + c].append(e)
        start = np.zeros(nx * ny + 1, np.int32)
        start[1:] = np.cumsum([len(x) for x in lists])
        items = np.asarray([e for x in lists for e in x] or [0], np.int32)
        return dict(boxes=np.asarray(self.boxes, np.float32).reshape(-1, BOX_WORDS),
                    hf_head=np.asarray(self.hf_head, np.float32).reshape(-1, HF_WORDS),
                    hf_int=np.asarray(self.hf_int, np.int32).reshape(-1, HF_INTS),
                    hf_data=np.asarray(self.hf_data, np.float32).reshape(-1),
                    grid=np.asarray([x0, y0, 1.0 / cell, nx, ny, nbox], np.float32),
                    cell_start=start, cell_items=items, elem_cell=first.reshape(-1, 2),
                    material=self.material_array(), sand=self.sand_array())


def _geom_ids(m, geoms):
    out = []
    for g in geoms:
        if isinstance(g, str):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if gid < 0:
                raise ValueError(f"TerrainConfig.geoms: no geom named {g!r} in the model")
        else:
            gid = int(g)
            if not 0 <= gid < m.ngeom:
                raise ValueError(f"TerrainConfig.geoms: geom id {gid} out of range (the model has {m.ngeom} geoms)")
        out.append(gid)
    return out


def robot_terrain_geoms(m, foot_pairs, which):
    """[(robot geom, its floor pair index, foot flag)] of the geoms that collide with the terrain: which = "feet" (the
    foot-floor pairs' robot geoms), "body" (every robot geom with a floor pair) or a list of geom names / ids (each
    with a floor pair), in floor-pair order."""
    foot = set(int(p) for p in foot_pairs)
    sel = None
    if not isinstance(which, str):
        sel = set(_geom_ids(m, which))
    elif which not in ("feet", "body"):
        raise ValueError(f"terrain geoms {which!r}: 'feet', 'body' or a list of geom names / ids")
    out = []
    for p in range(m.npair):
        g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        t1, t2 = int(m.geom_type[g1]), int(m.geom_type[g2])
        if mujoco.mjtGeom.mjGEOM_PLANE not in (t1, t2):
            continue
        g = g2 if t1 == mujoco.mjtGeom.mjGEOM_PLANE else g1
        if (sel is None and which == "feet" and p not in foot) or (sel is not None and g not in sel):
            continue
        if int(m.geom_type[g]) not in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                       mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError(f"terrain contacts of robot geom {m.geom(g).name or g} "
                             f"({mujoco.mjtGeom(int(m.geom_type[g])).name}): only spheres, capsules and boxes collide "
                             "with the terrain (TerrainConfig.geoms='feet' when only the feet are)")
        out.append((g, p, int(p in foot)))
    if sel is not None:
        missing = sel - {g for g, _, _ in out}
        if missing:
            raise ValueError("TerrainConfig.geoms: " + ", ".join(repr(m.geom(g).name or g) for g in sorted(missing))
                             + " without a floor pair (a terrain geom takes its floor pair's contact parameters)")
    return out
