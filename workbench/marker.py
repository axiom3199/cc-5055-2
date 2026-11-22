"""Virtual marker placement (in display coordinates) and per-frame rendering
of the solved scene overlay: map points, track tails, marker, and a HUD with
frame status. No camera smoothing is applied - the overlay directly reflects
the solved geometry."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import geometry as geo
from .obsgraph import ObsId


def make_cube_marker(center_xy, size=0.4, height=None):
    """Cube standing on the ground plane z=0 in display coordinates."""
    cx, cy = center_xy
    h = size if height is None else height
    s = size / 2.0
    V = np.array([
        [cx - s, cy - s, 0], [cx + s, cy - s, 0], [cx + s, cy + s, 0], [cx - s, cy + s, 0],
        [cx - s, cy - s, h], [cx + s, cy - s, h], [cx + s, cy + s, h], [cx - s, cy + s, h],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6),
             (1, 2, 6, 5), (0, 3, 7, 4)]
    return {"vertices": V, "edges": edges, "faces": faces,
            "spec": {"center_xy": [cx, cy], "size": size, "height": h}}


def _draw_marker(draw, uv, marker, color=(255, 80, 40)):
    for (a, b) in marker["edges"]:
        draw.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=2)
    # base quad translucent fill (contact with ground)
    base = [tuple(uv[i]) for i in (0, 1, 2, 3)]
    draw.polygon(base, outline=(255, 200, 60))


def render_frame_overlay(base_img, solution, frame_idx, K, marker=None,
                         display_transform=None, tracks=None, max_pts=4000):
    """Composite solved geometry over the original frame."""
    img = Image.fromarray(base_img).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    if frame_idx not in solution.cameras:
        draw.text((8, 8), f"frame {frame_idx}: UNSOLVED", fill=(255, 60, 60))
        return np.asarray(img)
    R, t = solution.cameras[frame_idx]

    # map points visible in this frame; outlier flags come from the residual
    # layer keyed by unified observation identity (strict lookup - a missing
    # residual is an error, not a default)
    res_store = solution.residual_store
    pts2d, cols = [], []
    for (pid, f, x, y), outl_legacy in zip(solution.observations, solution.obs_outlier):
        if f != frame_idx or pid not in solution.points:
            continue
        outl = res_store.get(ObsId(pid, f))[1] if res_store is not None else outl_legacy
        Xc = R @ solution.points[pid] + t
        if Xc[2] <= 0.1:
            continue
        uv = geo.project_points(Xc[None], K)[0]
        if 0 <= uv[0] < base_img.shape[1] and 0 <= uv[1] < base_img.shape[0]:
            pts2d.append(uv)
            if outl:
                cols.append((255, 40, 40, 255))
            elif solution.point_meta[pid].get("inherited"):
                cols.append((255, 170, 0, 255))
            else:
                cols.append((60, 255, 120, 255))
    for uv, c in list(zip(pts2d, cols))[:max_pts]:
        draw.ellipse([uv[0] - 2, uv[1] - 2, uv[0] + 2, uv[1] + 2], outline=c)

    if marker is not None and display_transform is not None:
        M = np.asarray(display_transform["matrix"], float)
        Minv = np.linalg.inv(M)
        V_raw = marker["vertices"] @ Minv[:3, :3].T + Minv[:3, 3]
        uv = geo.project_world(R, t, V_raw, K)
        _draw_marker(draw, uv, marker)

    info = solution.frame_info.get(frame_idx, {})
    status = info.get("method", "?")
    if info.get("rotation_only"):
        status += " [ROTATION-ONLY SEGMENT]"
    hud = f"frame {frame_idx}  status={status}  pts={len(pts2d)}"
    draw.rectangle([0, 0, 460, 18], fill=(0, 0, 0, 160))
    draw.text((6, 3), hud, fill=(255, 255, 255))
    return np.asarray(img)


def render_sequence(run_dir, solution, K, marker=None, display_transform=None,
                    frames_dir="video", out_dir="render"):
    from PIL import Image as PILImage
    src = os.path.join(run_dir, frames_dir)
    dst = os.path.join(run_dir, out_dir)
    os.makedirs(dst, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
    paths = []
    for i, fn in enumerate(files):
        base = np.asarray(PILImage.open(os.path.join(src, fn)).convert("L"))
        over = render_frame_overlay(base, solution, i, K, marker, display_transform)
        out = os.path.join(dst, fn)
        PILImage.fromarray(over).save(out)
        paths.append(out)
    return paths
