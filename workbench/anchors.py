"""Scale/ground anchors and lens projects.

An AnchorSet holds the user-supplied metric evidence for a reconstruction:
distance anchors (a known real distance between two map points) and ground
anchors (map points known to lie on the ground plane). The same AnchorSet -
in particular the same ground anchor - can back two lens projects over one
reconstruction.

A LensProject is one engineering view of the reconstruction:

  monocular  the only length reference is the distance anchor set. Deleting
             the last distance anchor suspends metric export: the preview
             (overlay render) keeps working, but landing points with units
             may no longer be exported.
  stereo     additionally carries a known rig baseline (a known real
             distance between two camera centers). Metric scale survives the
             deletion of every distance anchor, so metric landing points can
             still be exported.

Anchor deletion is an undoable graph transaction (see obsgraph.py), so it
races correctly with a background optimization commit.
"""
import numpy as np

from . import align
from . import geometry as geo


class LensKind:
    MONOCULAR = "monocular"
    STEREO = "stereo"


class MetricSuspendedError(RuntimeError):
    """Metric export is suspended: the project has no length reference.
    Preview is unaffected."""


class UnknownAnchorError(KeyError):
    pass


class AnchorSet:
    """Distance anchors + ground anchors for one reconstruction."""

    def __init__(self):
        self.distance_anchors = {}   # anchor_id -> {"points": (a, b), "meters": m}
        self.ground_ids = set()
        self._next_id = 0

    # -- distance anchors ---------------------------------------------------
    def reserve_id(self):
        """Reserve the id the next new anchor will get, without adding it.
        Lets a transaction stage an exact inverse before applying."""
        aid = self._next_id
        self._next_id += 1
        return aid

    def upsert_distance_anchor(self, points, meters, anchor_id=None):
        points = (int(points[0]), int(points[1]))
        if anchor_id is None:
            for aid, anc in self.distance_anchors.items():
                if tuple(anc["points"]) == points:
                    anchor_id = aid
                    break
        if anchor_id is None or anchor_id < 0:
            anchor_id = self._next_id
            self._next_id += 1
        self.distance_anchors[int(anchor_id)] = {
            "points": points, "meters": float(meters)}
        self._next_id = max(self._next_id, int(anchor_id) + 1)
        return int(anchor_id)

    def get_distance_anchor(self, anchor_id):
        anchor_id = int(anchor_id)
        if anchor_id not in self.distance_anchors:
            raise UnknownAnchorError(f"no distance anchor {anchor_id}")
        return self.distance_anchors[anchor_id]

    def remove_distance_anchor(self, anchor_id):
        return self.distance_anchors.pop(int(anchor_id))

    # -- ground anchors -----------------------------------------------------
    def add_ground(self, point_ids):
        self.ground_ids |= {int(p) for p in point_ids}

    def set_ground(self, point_ids):
        self.ground_ids = {int(p) for p in point_ids}

    # -- persistence --------------------------------------------------------
    def to_dict(self):
        return {"distance_anchors": [
                    {"id": int(aid), "points": list(anc["points"]),
                     "meters": anc["meters"]}
                    for aid, anc in sorted(self.distance_anchors.items())],
                "ground_ids": sorted(self.ground_ids),
                "next_id": self._next_id}

    @classmethod
    def from_dict(cls, d):
        a = cls()
        for anc in d.get("distance_anchors", []):
            a.upsert_distance_anchor(anc["points"], anc["meters"],
                                     anchor_id=anc["id"])
        a.set_ground(d.get("ground_ids", []))
        a._next_id = int(d.get("next_id", a._next_id))
        return a


class LensProject:
    """One lens-engineering view (monocular or stereo) over a shared
    observation graph. Projects of both kinds may wrap the SAME graph and
    thus the SAME ground anchor."""

    def __init__(self, kind, graph, baseline_m=None, baseline_pair=None):
        if kind not in (LensKind.MONOCULAR, LensKind.STEREO):
            raise ValueError(f"unknown lens kind {kind!r}")
        if kind == LensKind.STEREO:
            if baseline_m is None or baseline_pair is None:
                raise ValueError(
                    "a stereo project needs its known baseline: baseline_m "
                    "and baseline_pair=(frame_a, frame_b)")
            if float(baseline_m) <= 0:
                raise ValueError("baseline_m must be positive")
        self.kind = kind
        self.graph = graph
        self.baseline_m = None if baseline_m is None else float(baseline_m)
        self.baseline_pair = (None if baseline_pair is None
                              else (int(baseline_pair[0]), int(baseline_pair[1])))

    # -- metric evidence ------------------------------------------------------
    def scale_evidence(self, solution):
        """Where does metric scale come from right now?
        A live distance anchor first; otherwise, for a stereo project, the
        known rig baseline. None means the project has no length reference."""
        for aid, anc in sorted(self.graph.anchors.distance_anchors.items()):
            if anc["points"][0] not in solution.points or \
                    anc["points"][1] not in solution.points:
                continue   # anchor points deleted from the solution: no evidence
            pa = np.asarray(solution.points[anc["points"][0]], float)
            pb = np.asarray(solution.points[anc["points"][1]], float)
            raw = float(np.linalg.norm(pa - pb))
            if raw < 1e-12:
                continue
            return {"kind": "distance_anchor", "anchor_id": int(aid),
                    "scale": anc["meters"] / raw, "meters": anc["meters"]}
        if self.kind == LensKind.STEREO:
            fa, fb = self.baseline_pair
            Ca = geo.camera_center(*solution.cameras[fa])
            Cb = geo.camera_center(*solution.cameras[fb])
            raw = float(np.linalg.norm(Ca - Cb))
            if raw < 1e-12:
                raise MetricSuspendedError(
                    "stereo baseline cameras coincide in the solution")
            return {"kind": "stereo_baseline", "scale": self.baseline_m / raw,
                    "meters": self.baseline_m}
        return None

    def metric_available(self, solution):
        return self.scale_evidence(solution) is not None

    # -- metric transform + landing export ------------------------------------
    def metric_transform(self, solution):
        """Display transform (ground plane -> z=0, +Z up) at metric scale.
        Raises MetricSuspendedError when no length reference remains."""
        ev = self.scale_evidence(solution)
        if ev is None:
            raise MetricSuspendedError(
                f"{self.kind} project has no length reference: metric export "
                f"is suspended (preview is unaffected)")
        ground = sorted(self.graph.anchors.ground_ids)
        if not ground:
            raise MetricSuspendedError(
                "no ground anchor: landing points are undefined")
        missing = [g for g in ground if g not in solution.points]
        if missing:
            raise MetricSuspendedError(
                f"ground anchor points {missing} are not in the current "
                f"solution (deleted or pruned); re-identify the ground")
        gnd = np.array([solution.points[g] for g in ground], float)
        frames = sorted(solution.cameras)
        centers = np.array([geo.camera_center(*solution.cameras[f]) for f in frames])
        R_align, t_align = align.orientation_from_ground(gnd, centers, ev["scale"])
        M = geo.similarity_matrix(ev["scale"], R_align, t_align)
        return {"matrix": M.tolist(), "scale": ev["scale"],
                "rotation": R_align.tolist(), "translation": t_align.tolist(),
                "scale_evidence": ev["kind"],
                "ground_plane_rms_m": float(np.sqrt(np.mean(
                    (gnd @ R_align.T * ev["scale"] + t_align)[:, 2] ** 2)))}

    def export_landing_points(self, solution, z_tol_m=0.10):
        """Ground-contact ('landing') points in meters. Suspended without a
        length reference; the overlay preview keeps working regardless."""
        tf = self.metric_transform(solution)   # raises when suspended
        M = np.asarray(tf["matrix"], float)
        tol = max(float(z_tol_m), 2.0 * tf["ground_plane_rms_m"])
        out = []
        for pid, X in sorted(solution.points.items()):
            Xd = X @ M[:3, :3].T + M[:3, 3]
            if abs(Xd[2]) <= tol:
                out.append({"id": int(pid), "xyz_m": [float(v) for v in Xd]})
        return {"units": "m", "scale_evidence": tf["scale_evidence"],
                "z_tolerance_m": tol,
                "ground_plane_rms_m": tf["ground_plane_rms_m"],
                "points": out}
