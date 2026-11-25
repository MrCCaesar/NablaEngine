"""Sensor definitions: mount (body, position, roll-pitch-yaw in the body frame, as a URDF fixed joint), ray pattern in
the sensor frame (x forward, y left, z up: the URDF link convention), range limits. The G1's sensors from Unitree's
URDF (unitree_ros robots/g1_description/g1_29dof_rev_1_0.urdf: d435_joint, mid360_joint on torso_link; the mjlab G1's
torso_link frame is checked to be the URDF's). Field of view and ranges after the vendors' datasheets (to be
checked against them: the requirements note marks them as from memory)."""
from dataclasses import dataclass, replace

import numpy as np


def rpy_matrix(r, p, y):
    """URDF rpy: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


INTRINSICS_NOISE_MAX = 0.2


def pixel_dirs(xp, width, height, fx, fy, cx, cy):
    """Unit ray directions (..., H W, 3) in the sensor frame (x forward, y left, z up) of a pinhole camera with focal
    lengths fx, fy and principal point cx, cy in pixels (u right from the image's left edge, v down from its top edge;
    pixel centres at + 0.5); fx ... cy scalars or arrays (...) broadcast together (xp: numpy or jax.numpy)."""
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5, indexing="xy")
    u, v = xp.asarray(u.reshape(-1)), xp.asarray(v.reshape(-1))
    fx, fy, cx, cy = (xp.asarray(a)[..., None] for a in (fx, fy, cx, cy))
    y, z = -(u - cx) / fx, -(v - cy) / fy
    d = xp.stack([xp.ones_like(y + z), y + 0 * z, z + 0 * y], -1)
    return d / xp.linalg.norm(d, axis=-1, keepdims=True)


@dataclass(frozen=True)
class DepthCamera:
    """A pinhole depth camera: width x height pixels over hfov x vfov degrees (pixel centres), z-depth along the optical
    axis (sensor x), valid in [min_depth, max_depth]. fx, fy, cx, cy (pixels, optional): the intrinsics in place of
    hfov / vfov and the centred principal point (each one given replaces its default: fx = width / 2 / tan(hfov / 2),
    cx = width / 2, ...; all None, the default, is the field-of-view camera, bitwise as before). quat (optional, w x y
    z): the mount's rotation in the body frame in place of rpy (e.g. from_mjcf's)."""
    body: str
    pos: tuple
    rpy: tuple
    width: int = 87
    height: int = 58
    hfov: float = 87.0
    vfov: float = 58.0
    min_depth: float = 0.28
    max_depth: float = 10.0
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None
    quat: tuple = None

    def with_resolution(self, width, height):
        """The camera at width x height pixels (explicit fx, fy, cx, cy are scaled with the image)."""
        sx, sy = int(width) / self.width, int(height) / self.height
        sc = lambda a, s: None if a is None else float(a) * s
        return replace(self, width=int(width), height=int(height), fx=sc(self.fx, sx), fy=sc(self.fy, sy),
                       cx=sc(self.cx, sx), cy=sc(self.cy, sy))

    def with_intrinsics(self, fx=None, fy=None, cx=None, cy=None):
        """The camera with these intrinsics in pixels (None: keep the current value or default)."""
        return replace(self, fx=self.fx if fx is None else float(fx), fy=self.fy if fy is None else float(fy),
                       cx=self.cx if cx is None else float(cx), cy=self.cy if cy is None else float(cy))

    def intrinsics(self):
        """(fx, fy, cx, cy) in pixels (Python floats)."""
        fx = self.width / 2 / np.tan(np.radians(self.hfov) / 2) if self.fx is None else self.fx
        fy = self.height / 2 / np.tan(np.radians(self.vfov) / 2) if self.fy is None else self.fy
        return (float(fx), float(fy), float(self.width / 2 if self.cx is None else self.cx),
                float(self.height / 2 if self.cy is None else self.cy))

    def explicit(self):
        """True where any of fx, fy, cx, cy is given (rays() then uses pixel_dirs)."""
        return any(a is not None for a in (self.fx, self.fy, self.cx, self.cy))

    def noisy_intrinsics(self, noise):
        """(fx, fy, cx, cy) per world (B,) each from noise (B, 4) = (relative change of fx, of fy, shift of cx as a
        fraction of the width, of cy as a fraction of the height), each clipped to +-INTRINSICS_NOISE_MAX (JAX)."""
        import jax.numpy as jnp
        fx, fy, cx, cy = self.intrinsics()
        e = jnp.clip(jnp.asarray(noise, jnp.float32), -INTRINSICS_NOISE_MAX, INTRINSICS_NOISE_MAX)
        return fx * (1 + e[:, 0]), fy * (1 + e[:, 1]), cx + e[:, 2] * self.width, cy + e[:, 3] * self.height

    def noisy_cone_cos(self):
        """The smallest optical-axis cosine of any ray of the camera under any intrinsics noise within
        INTRINSICS_NOISE_MAX (the corners of the image at the extremes)."""
        fx, fy, cx, cy = self.intrinsics()
        a = INTRINSICS_NOISE_MAX
        worst = 1.0
        for sx in (1 - a, 1 + a):
            for sy in (1 - a, 1 + a):
                for dcx in (-a, a):
                    for dcy in (-a, a):
                        d = pixel_dirs(np, self.width, self.height, fx * sx, fy * sy, cx + dcx * self.width,
                                       cy + dcy * self.height)
                        worst = min(worst, float(d[:, 0].min()))
        return worst

    def rays(self):
        """(offsets (N, 3) zeros, unit directions (N, 3)) in the sensor frame, row-major pixels (v down, u right)."""
        if self.explicit():
            d = pixel_dirs(np, self.width, self.height, *(np.float64(x) for x in self.intrinsics()))
            return np.zeros_like(d, dtype=np.float32), d.astype(np.float32)
        fx = self.width / 2 / np.tan(np.radians(self.hfov) / 2)
        fy = self.height / 2 / np.tan(np.radians(self.vfov) / 2)
        u, v = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5, indexing="xy")
        d = np.stack([np.ones_like(u), -(u - self.width / 2) / fx, -(v - self.height / 2) / fy], -1).reshape(-1, 3)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return np.zeros_like(d, dtype=np.float32), d.astype(np.float32)

    def mount(self):
        q = getattr(self, "quat", None)
        return np.asarray(self.pos, np.float64), (rpy_matrix(*self.rpy) if q is None
                                                  else quat_matrix(np.asarray(q, np.float64) / np.linalg.norm(q)))

    def pose(self, body_pos, body_quat):
        """World position and rotation (3, 3) of the sensor for a body pose (NumPy, one world)."""
        mp, mR = self.mount()
        R = quat_matrix(body_quat)
        return body_pos + R @ mp, R @ mR

    def depth(self, dist, dirs):
        """z-depth (..., N) from ray distances along unit dirs (sensor frame): dist * dir_x; invalid (miss, nearer than
        min_depth, beyond max_depth) as 0, the RealSense convention."""
        z = dist * dirs[..., 0]
        ok = (dist >= 0) & (z >= self.min_depth) & (z <= self.max_depth)
        return z * ok


MJ_CAM_TO_SENSOR = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def from_mjcf(model, camera, width, height, min_depth=0.0, max_depth=20.0):
    """A DepthCamera from an MJCF <camera> of model (a name or id; a fixed camera: mode "fixed"): its body, position
    and orientation (MuJoCo's convention: looking along the camera's -z, y up, x right), at width x height pixels.
    Intrinsics as MuJoCo's renderer: fovy is the vertical field of view, square pixels (fx = fy = height / 2 /
    tan(fovy / 2)), the principal point at the image centre; a camera with sensorsize / focal / principal: fx =
    focal_x / sensorsize_x width, fy = focal_y / sensorsize_y height, cx = width / 2 - principal_x / sensorsize_x
    width, cy = height / 2 - principal_y / sensorsize_y height (pixels from the image's left / top edge; both forms
    checked against MuJoCo's renderer's segmentation)."""
    import mujoco
    c = int(camera) if not isinstance(camera, str) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if not 0 <= c < model.ncam:
        raise ValueError(f"from_mjcf: no camera {camera!r}")
    if int(model.cam_mode[c]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
        raise ValueError(f"from_mjcf: camera {camera!r} is not mode='fixed' (tracking cameras are not supported)")
    b = int(model.cam_bodyid[c])
    if b == 0:
        raise ValueError(f"from_mjcf: camera {camera!r} is on the world body (a body-mounted camera is needed)")
    R = quat_matrix(np.asarray(model.cam_quat[c], np.float64)) @ MJ_CAM_TO_SENSOR
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    ss = np.asarray(model.cam_sensorsize[c], np.float64)
    if ss[0] > 0 and ss[1] > 0:
        f = np.asarray(model.cam_intrinsic[c], np.float64)
        fx, fy = f[0] / ss[0] * width, f[1] / ss[1] * height
        cx, cy = width / 2 - f[2] / ss[0] * width, height / 2 - f[3] / ss[1] * height
    else:
        fy = height / 2 / np.tan(np.radians(float(model.cam_fovy[c])) / 2)
        fx, cx, cy = fy, width / 2, height / 2
    hfov = float(np.degrees(2 * np.arctan(width / 2 / fx)))
    vfov = float(np.degrees(2 * np.arctan(height / 2 / fy)))
    return DepthCamera(body=model.body(b).name, pos=tuple(float(x) for x in model.cam_pos[c]), rpy=(0.0, 0.0, 0.0),
                       width=int(width), height=int(height), hfov=hfov, vfov=vfov, min_depth=float(min_depth),
                       max_depth=float(max_depth), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                       quat=tuple(float(x) for x in q))


D435_STREAMS = {
    "1280x720": (87.0, 58.0, 0.280), "848x480": (87.0, 58.0, 0.195), "640x480": (75.0, 62.0, 0.175),
    "640x360": (87.0, 58.0, 0.150), "480x270": (87.0, 58.0, 0.120), "424x240": (87.0, 58.0, 0.105),
}
D435I_DEPTH = DepthCamera(body="torso_link", pos=(0.0576235, 0.01753, 0.42987), rpy=(0.0, 0.8307767239493009, 0.0))


D435I_COLOR = DepthCamera(body="torso_link", pos=tuple(np.asarray(D435I_DEPTH.pos) + rpy_matrix(*D435I_DEPTH.rpy)
                                                        @ np.array([0.0, 0.015, 0.0])),
                          rpy=D435I_DEPTH.rpy, width=64, height=36, hfov=69.0, vfov=42.0, min_depth=0.0, max_depth=20.0)


def d435i_depth(stream="1280x720", width=None, height=None):
    """D435I_DEPTH with the field of view and minimum depth of the real camera's depth stream (D435_STREAMS; the
    minimum depth shrinks with the stream's resolution), rendered at width x height rays (default: the default
    DepthCamera's)."""
    if stream not in D435_STREAMS:
        raise ValueError(f"d435i_depth: stream {stream!r}: one of {sorted(D435_STREAMS)}")
    hfov, vfov, zmin = D435_STREAMS[stream]
    cam = replace(D435I_DEPTH, hfov=hfov, vfov=vfov, min_depth=zmin)
    return cam.with_resolution(width, height) if width is not None else cam


# Livox's recorded MID-360 scan (Livox-SDK/livox_laser_simulation scan_mode/mid360.csv, MIT license: data/
# LIVOX_MID360_LICENSE): 800000 consecutive firing directions (azimuth, zenith in degrees, sensor frame) at the
# sensor's 200000 points per second, i.e. 4 s of its non-repetitive pattern
MID360_PATTERN = "livox_mid360.npz"
MID360_POINT_RATE = 200000


def livox_pattern(name=MID360_PATTERN):
    """(azimuth, zenith) in radians of a packaged Livox scan pattern."""
    from pathlib import Path
    z = np.load(Path(__file__).parent / "data" / name)
    return np.radians(z["azimuth_deg"].astype(np.float64)), np.radians(z["zenith_deg"].astype(np.float64))


@dataclass(frozen=True)
class Lidar:
    """A LiDAR's rays per frame (rate_hz frames per second), returns within [min_range, max_range]. pattern "mid360":
    Livox's recorded MID-360 scan (livox_pattern): frame f is the f-th 1 / rate_hz of firing directions (20000 at 10 Hz,
    covering azimuth 0 .. 360 deg and elevation -7 .. 52 deg, different every frame; frames() gives the 4 s / (1 /
    rate_hz) = 40 frames, each thinned evenly to n rays); pattern "spiral": one fixed frame, n rays spread evenly over
    azimuth and elevation vmin .. vmax deg (a golden-angle spiral, uniform in solid angle)."""
    body: str
    pos: tuple
    rpy: tuple
    n: int = 4096
    vmin: float = -7.0
    vmax: float = 52.0
    min_range: float = 0.1
    max_range: float = 40.0
    rate_hz: float = 10.0
    pattern: str = "spiral"

    def with_rays(self, n):
        return replace(self, n=int(n))

    def frames(self):
        """Unit ray directions (F, n, 3) in the sensor frame, one row per frame (F = 1 for "spiral")."""
        if self.pattern == "spiral":
            k = np.arange(self.n) + 0.5
            s0, s1 = np.sin(np.radians(self.vmin)), np.sin(np.radians(self.vmax))
            el = np.arcsin(s0 + (s1 - s0) * k / self.n)
            az = (k * np.pi * (3 - np.sqrt(5))) % (2 * np.pi)
            return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)[None].astype(np.float32)
        if self.pattern != "mid360":
            raise ValueError(f"Lidar pattern {self.pattern!r}: 'mid360' or 'spiral'")
        az, ze = livox_pattern()
        per = int(round(MID360_POINT_RATE / self.rate_hz))
        if not 0 < self.n <= per:
            raise ValueError(f"Lidar: {self.n} rays per frame, the MID-360 fires {per} per frame at {self.rate_hz} Hz")
        F = len(az) // per
        idx = (np.arange(F)[:, None] * per + (np.arange(self.n) * per) // self.n)
        a, z = az[idx], ze[idx]
        return np.stack([np.sin(z) * np.cos(a), np.sin(z) * np.sin(a), np.cos(z)], -1).astype(np.float32)

    def rays(self):
        """(offsets (n, 3) zeros, unit directions (n, 3)) of frame 0."""
        d = self.frames()[0]
        return np.zeros_like(d), d

    mount = DepthCamera.mount
    pose = DepthCamera.pose


@dataclass(frozen=True)
class HeightScan:
    """mjlab's RayCastSensor height scan: a grid of size (x, y) metres at resolution, rays straight down from offset_z
    above the body frame, yaw-aligned (mjlab GridPatternCfg, ray_alignment "yaw"); the output is the body height above
    each hit (pos_z - hit_z), as mjlab's height_scan observation."""
    body: str = "pelvis"
    size: tuple = (1.6, 1.0)
    resolution: float = 0.1
    offset_z: float = 0.5
    max_distance: float = 5.0

    def rays(self):
        x = np.arange(-self.size[0] / 2, self.size[0] / 2 + self.resolution * 0.5, self.resolution)
        y = np.arange(-self.size[1] / 2, self.size[1] / 2 + self.resolution * 0.5, self.resolution)
        gx, gy = np.meshgrid(x, y, indexing="xy")
        off = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, self.offset_z)], -1).astype(np.float32)
        return off, np.tile(np.float32([0, 0, -1]), (len(off), 1))


MID360 = Lidar(body="torso_link", pos=(0.0002835, 0.00003, 0.428434), rpy=(np.pi, 0.05112069379091391, 0.0),
               pattern="mid360")
