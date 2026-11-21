"""Procedural test footage: textured ground plane (ray-cast) + dotted boxes,
camera trajectory with pure-rotation segments, and ground truth."""
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import geometry as geo


@dataclass
class SynthConfig:
    width: int = 640
    height: int = 480
    fx: float = 700.0
    seed: int = 7
    profile: str = "survey"          # 'survey' (translation + pure rotation) or 'translate_only'
    n_frames: int = 100
    noise_sigma: float = 1.2
    n_ground_dots: int = 150
    n_box_points: int = 60

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.width / 2.0],
                         [0.0, self.fx, self.height / 2.0],
                         [0.0, 0.0, 1.0]])


def make_ground_texture(seed, size=1024, extent=14.0):
    """Random multi-scale blob texture for the ground plane."""
    rng = np.random.default_rng(seed + 100)
    tex = np.full((size, size), 60.0)
    yy, xx = np.mgrid[0:size, 0:size]
    for scale, amp in [(4, 45), (8, 38), (16, 30), (32, 24), (64, 16), (128, 10), (256, 6)]:
        g = rng.normal(0, 1, (scale + 1, scale + 1))
        gx = np.linspace(0, scale, size)
        gy = np.linspace(0, scale, size)
        xi = np.clip(np.searchsorted(np.arange(scale + 1), gx) - 1, 0, scale - 1)
        yi = np.clip(np.searchsorted(np.arange(scale + 1), gy) - 1, 0, scale - 1)
        fx_ = (gx - xi)[:, None]
        fy_ = (gy - yi)[:, None]
        g00 = g[np.ix_(yi, xi)]
        g01 = g[np.ix_(yi, xi + 1)]
        g10 = g[np.ix_(yi + 1, xi)]
        g11 = g[np.ix_(yi + 1, xi + 1)]
        tex += amp * ((g00 * (1 - fx_) + g01 * fx_) * (1 - fy_).T
                      + (g10 * (1 - fx_) + g11 * fx_) * fy_.T)
    # scattered pebbles with crisp edges
    for _ in range(700):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(2, 8)
        c = rng.uniform(-70, 110)
        yy0, xx0 = np.ogrid[:size, :size]
        m = (xx0 - cx) ** 2 + (yy0 - cy) ** 2 <= r * r
        tex[m] += c
    tex = np.clip(tex, 5, 250)
    return tex, extent


def make_scene(cfg: SynthConfig):
    rng = np.random.default_rng(cfg.seed)
    gxy = rng.uniform(-9.0, 9.0, size=(cfg.n_ground_dots, 2))
    ground = np.column_stack([gxy, np.zeros(cfg.n_ground_dots)])
    box_specs = [(2.0, 1.0, 1.6, 1.2, 2.2), (-3.0, 2.0, 1.2, 1.2, 1.6),
                 (4.0, -3.0, 1.5, 1.5, 2.8), (-2.0, -4.0, 1.8, 1.4, 1.9),
                 (0.5, 5.0, 1.3, 1.3, 2.4), (-5.5, -1.5, 1.4, 1.4, 2.0)]
    box_pts = []
    for (cx, cy, sx, sy, h) in box_specs:
        n = cfg.n_box_points
        face = rng.integers(0, 4, n)
        a = rng.uniform(-0.5, 0.5, n)
        z = rng.uniform(0.0, h, n)
        pts = np.zeros((n, 3))
        for i in range(n):
            if face[i] == 0:
                pts[i] = [cx + a[i] * sx, cy - sy / 2, z[i]]
            elif face[i] == 1:
                pts[i] = [cx + a[i] * sx, cy + sy / 2, z[i]]
            elif face[i] == 2:
                pts[i] = [cx - sx / 2, cy + a[i] * sy, z[i]]
            else:
                pts[i] = [cx + sx / 2, cy + a[i] * sy, z[i]]
        box_pts.append(pts)
    boxes = np.vstack(box_pts)
    points = np.vstack([ground, boxes])
    ground_mask = np.zeros(len(points), bool)
    ground_mask[:len(ground)] = True
    colors = rng.integers(150, 256, size=len(points)).astype(np.uint8)
    size_jitter = rng.uniform(0.8, 1.5, size=len(points))
    pattern_angle = rng.uniform(0, 360, size=len(points))
    return points, colors, ground_mask, box_specs, size_jitter, pattern_angle


def _look_at(C, target, world_up=(0, 0, 1)):
    fwd = np.asarray(target, float) - C
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(world_up, float)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.column_stack([right, down, fwd])  # cam -> world


def make_trajectory(cfg: SynthConfig):
    n = cfg.n_frames
    frames = []
    base_radius, base_height = 10.0, 3.2
    ang_start, ang_mid, ang_end = -50.0, -28.0, -12.0

    def arc_pos(deg):
        """Position on a gently varying arc; radius/height are functions of the
        angle only, so the path is continuous through the rotation segments."""
        v = (deg - ang_start) / (ang_end - ang_start)
        radius = base_radius + 1.6 * np.sin(2 * np.pi * v)
        height = base_height + 0.9 * np.sin(2 * np.pi * v + 1.0)
        a = np.radians(deg)
        return np.array([radius * np.sin(a), -radius * np.cos(a), height])

    if cfg.profile == "survey":
        b1, b2, b3 = int(0.4 * n), int(0.6 * n), int(0.8 * n)
        for f in range(n):
            if f < b1:
                ang = ang_start + (ang_mid - ang_start) * f / max(b1 - 1, 1)
                C = arc_pos(ang)
                R_cw = _look_at(C, (0, 0, 0.8))
                seg = "translate"
            elif f < b2:
                C = arc_pos(ang_mid)   # frozen: pure rotation
                k = (f - b1) / max(b2 - b1 - 1, 1)
                yaw = np.radians(14.0) * k
                pitch = np.radians(4.0) * np.sin(k * np.pi)
                R0 = _look_at(C, (0, 0, 0.8))
                R_cw = geo.rodrigues(np.array([0.0, 0.0, yaw])) @ R0 @ geo.rodrigues(np.array([pitch, 0.0, 0.0]))
                seg = "rotate"
            elif f < b3:
                k = (f - b2) / max(b3 - b2 - 1, 1)
                ang = ang_mid + (ang_end - ang_mid) * k
                C = arc_pos(ang)
                R0 = _look_at(C, (0, 0, 0.8))
                R_cw = geo.rodrigues(np.array([0.0, 0.0, np.radians(14.0)])) @ R0
                seg = "translate"
            else:
                C = arc_pos(ang_end)   # frozen: pure rotation
                k = (f - b3) / max(n - b3 - 1, 1)
                yaw = np.radians(14.0) + np.radians(20.0) * k
                R0 = _look_at(C, (0, 0, 0.8))
                R_cw = geo.rodrigues(np.array([0.0, 0.0, yaw])) @ R0
                seg = "rotate"
            frames.append({"C": C, "R_cw": R_cw, "segment": seg})
    elif cfg.profile == "translate_only":
        for f in range(n):
            ang = ang_start + (ang_end - ang_start) * f / max(n - 1, 1)
            C = arc_pos(ang)
            R_cw = _look_at(C, (0, 0, 0.8))
            frames.append({"C": C, "R_cw": R_cw, "segment": "translate"})
    else:
        raise ValueError(f"unknown profile {cfg.profile}")
    return frames


def _render_ground(cfg, tex, extent, C, R_cw):
    """Ray-cast the textured ground plane z=0 for every pixel."""
    h, w = cfg.height, cfg.width
    size = tex.shape[0]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    rays_cam = np.stack([(us - cfg.K[0, 2]) / cfg.K[0, 0],
                         (vs - cfg.K[1, 2]) / cfg.K[1, 1],
                         np.ones_like(us)], axis=-1)  # (h,w,3)
    rays_w = rays_cam @ R_cw.T
    rz = rays_w[..., 2]
    t = np.where(np.abs(rz) > 1e-9, -C[2] / rz, np.inf)
    valid = (t > 0) & np.isfinite(t)
    hx = C[0] + t * rays_w[..., 0]
    hy = C[1] + t * rays_w[..., 1]
    inside = valid & (np.abs(hx) < extent) & (np.abs(hy) < extent)
    tx = (hx + extent) / (2 * extent) * (size - 1)
    ty = (hy + extent) / (2 * extent) * (size - 1)
    x0 = np.clip(np.floor(tx).astype(int), 0, size - 2)
    y0 = np.clip(np.floor(ty).astype(int), 0, size - 2)
    fx_ = np.clip(tx - x0, 0, 1)
    fy_ = np.clip(ty - y0, 0, 1)
    img = (tex[y0, x0] * (1 - fx_) * (1 - fy_) + tex[y0, x0 + 1] * fx_ * (1 - fy_)
           + tex[y0 + 1, x0] * (1 - fx_) * fy_ + tex[y0 + 1, x0 + 1] * fx_ * fy_)
    # distance attenuation
    dist = np.where(inside, t, 1.0)
    shade = np.clip(1.4 - dist / 40.0, 0.55, 1.0)
    out = np.where(inside, img * shade, 12.0)
    return out


def render_frame(cfg, points, colors, size_jitter, pattern_angle, tex, extent, C, R_cw, rng):
    base = _render_ground(cfg, tex, extent, C, R_cw)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "L")
    draw = ImageDraw.Draw(img)
    R_wc = R_cw.T
    t = -R_wc @ C
    Xc = geo.world_to_cam(R_wc, t, points)
    z = Xc[:, 2]
    uv = geo.project_points(Xc, cfg.K)
    vis = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < cfg.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cfg.height)
    order = np.argsort(-z)
    for i in order:
        if not vis[i]:
            continue
        r = float(np.clip(2.6 * 8.0 / z[i], 1.4, 3.8)) * size_jitter[i]
        x, y = uv[i]
        c = int(colors[i])
        bbox = [x - r, y - r, x + r, y + r]
        a = float(pattern_angle[i])
        draw.ellipse(bbox, fill=c)
        draw.pieslice(bbox, start=a, end=a + 180.0, fill=max(0, c - 110))
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    arr = np.asarray(img).astype(np.float64)
    if cfg.noise_sigma > 0:
        arr += rng.normal(0.0, cfg.noise_sigma, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def generate(out_dir, cfg: SynthConfig):
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "video")
    os.makedirs(frames_dir, exist_ok=True)
    points, colors, ground_mask, box_specs, size_jitter, pattern_angle = make_scene(cfg)
    tex, extent = make_ground_texture(cfg.seed)
    traj = make_trajectory(cfg)
    rng = np.random.default_rng(cfg.seed + 1)
    gt_frames = []
    for f, fr in enumerate(traj):
        img = render_frame(cfg, points, colors, size_jitter, pattern_angle,
                           tex, extent, fr["C"], fr["R_cw"], rng)
        Image.fromarray(img).save(os.path.join(frames_dir, f"frame_{f:04d}.png"))
        R_wc = fr["R_cw"].T
        t = -R_wc @ fr["C"]
        c2w = np.eye(4)
        c2w[:3, :3] = fr["R_cw"]
        c2w[:3, 3] = fr["C"]
        gt_frames.append({
            "idx": f,
            "segment": fr["segment"],
            "cam_to_world": c2w.reshape(-1).tolist(),
            "R_wc": R_wc.reshape(-1).tolist(),
            "t_wc": t.tolist(),
        })
    gt = {
        "K": cfg.K.tolist(),
        "image": [cfg.width, cfg.height],
        "profile": cfg.profile,
        "frames": gt_frames,
        "points": points.tolist(),
        "ground_mask": ground_mask.tolist(),
        "box_specs": [list(b) for b in box_specs],
        "ground_plane": [0.0, 0.0, 1.0, 0.0],
    }
    with open(os.path.join(out_dir, "gt.json"), "w") as fh:
        json.dump(gt, fh)
    return gt
