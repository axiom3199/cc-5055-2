"""Layered observation graph with undoable edit transactions.

Three layers are kept apart and share one observation identity:

  TrackStore     feature trajectories (storage): track_id -> frames/xy
  CameraStore    camera parameters (storage):    frame -> (R, t)
  ResidualStore  optimization residuals (evaluation): ObsId -> (px, outlier)

Every observation in the system is identified by an ObsId(track_id, frame);
the same key addresses the track layer (pixel position) and the residual
layer (post-BA reprojection error), so storage and evaluation can evolve
separately without losing the join.

Edits never mutate the graph in place: an Edit transaction is opened against
the current version, records its inverse as it goes, and commits only if the
graph is still at the version it was opened on. A background optimization
commit and a user delete therefore cannot both land on the same base - the
second committer is rejected with StaleEditError and must re-open. The
committed log is persisted as edits.jsonl and re-validated on load, so a
reopened project contains exactly the legal commits, in order.

The solver receives a read-only ObservationView snapshot; it cannot reach
the live graph and therefore cannot modify the UI-visible point set.
"""
import json
import os
from collections import namedtuple

import numpy as np

from .anchors import AnchorSet


class ObsId(namedtuple("ObsId", ["track_id", "frame"])):
    """Unified observation identity: one (track, frame) pair, stable across
    the storage and evaluation layers and across track splits (a fragment
    keeps its member ObsIds; only its track-level grouping changes)."""
    __slots__ = ()

    def __str__(self):
        return f"{self.track_id}@{self.frame}"


class UnknownObservationError(KeyError):
    pass


class StaleEditError(RuntimeError):
    """The graph moved on since this edit was opened (e.g. a background
    optimization committed first). Re-open the edit and retry."""


class ReidentifyRequired(RuntimeError):
    """Undo cannot recover the original constraint automatically: the track
    was split and its members no longer union cleanly to the original. The
    user must re-identify which fragments belong together. Binding by track
    name alone is never attempted."""


class CorruptEditLogError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# layer 1: feature tracks (storage)
# --------------------------------------------------------------------------
class TrackRecord:
    __slots__ = ("id", "frames", "xy", "origin", "lineage")

    def __init__(self, tid, frames, xy, origin=None, lineage=()):
        self.id = int(tid)
        self.frames = np.asarray(frames, np.int32)
        self.xy = np.asarray(xy, np.float32).reshape(-1, 2)
        if len(self.frames) != len(self.xy):
            raise ValueError("frames/xy length mismatch")
        # origin: the track this record descends from (itself if unsplit)
        self.origin = self.id if origin is None else int(origin)
        # lineage: the chain of retired ancestor track ids, nearest last;
        # a track is a descendant of every id in its lineage
        self.lineage = tuple(int(t) for t in lineage)

    def members(self):
        return {ObsId(self.origin, int(f)) for f in self.frames}

    def copy(self):
        return TrackRecord(self.id, self.frames.copy(), self.xy.copy(),
                           self.origin, self.lineage)


class TrackStore:
    def __init__(self):
        self._tracks = {}
        self._next_id = 0

    def add(self, record):
        if record.id in self._tracks:
            raise ValueError(f"track {record.id} already exists")
        self._tracks[record.id] = record
        self._next_id = max(self._next_id, record.id + 1)

    def get(self, tid):
        # strict: a missing track is an error, never a default
        if tid not in self._tracks:
            raise UnknownObservationError(f"no track {tid}")
        return self._tracks[tid]

    def ids(self):
        return sorted(self._tracks)

    def __contains__(self, tid):
        return tid in self._tracks

    def __len__(self):
        return len(self._tracks)

    def remove(self, tid):
        return self._tracks.pop(tid)

    def fresh_id(self):
        nid = self._next_id
        while nid in self._tracks:
            nid += 1
        return nid

    def split(self, tid, cut_frames):
        """Remove the observations at cut_frames from track tid and
        re-register the remaining contiguous run(s) as fragments with FRESH
        ids. The original id is retired (never reused for a fragment) so a
        later lookup by name cannot silently bind to a different track.
        Returns (removed, fragments)."""
        rec = self.get(tid)
        cut = set(int(f) for f in cut_frames)
        keep = [i for i, f in enumerate(rec.frames) if int(f) not in cut]
        gone = [i for i, f in enumerate(rec.frames) if int(f) in cut]
        removed = [(int(rec.frames[i]), rec.xy[i].copy()) for i in gone]
        self.remove(tid)
        fragments = []
        runs, cur = [], []
        for i in keep:
            if cur and int(rec.frames[i]) != int(rec.frames[cur[-1]]) + 1:
                runs.append(cur)
                cur = []
            cur.append(i)
        if cur:
            runs.append(cur)
        for run in runs:
            frag = TrackRecord(self.fresh_id(), rec.frames[run], rec.xy[run],
                               origin=rec.origin,
                               lineage=rec.lineage + (rec.id,))
            self.add(frag)
            fragments.append(frag)
        return removed, fragments

    def snapshot(self):
        """Deep copies, so the solver can never mutate the live store."""
        return [self._tracks[t].copy() for t in self.ids()]


# --------------------------------------------------------------------------
# layer 2: camera parameters (storage)
# --------------------------------------------------------------------------
class CameraStore:
    def __init__(self):
        self._cams = {}

    def set(self, frame, R, t):
        self._cams[int(frame)] = (np.asarray(R, float).reshape(3, 3),
                                  np.asarray(t, float).reshape(3))

    def get(self, frame):
        frame = int(frame)
        if frame not in self._cams:
            raise UnknownObservationError(f"no camera for frame {frame}")
        return self._cams[frame]

    def frames(self):
        return sorted(self._cams)

    def __contains__(self, frame):
        return int(frame) in self._cams

    def __len__(self):
        return len(self._cams)

    def as_dict(self):
        return {f: (R.copy(), t.copy()) for f, (R, t) in self._cams.items()}


# --------------------------------------------------------------------------
# layer 3: optimization residuals (evaluation)
# --------------------------------------------------------------------------
class ResidualStore:
    def __init__(self):
        self._res = {}

    def set(self, obs_id, residual_px, outlier=False):
        self._res[ObsId(int(obs_id.track_id), int(obs_id.frame))] = (
            float(residual_px), bool(outlier))

    def get(self, obs_id):
        obs_id = ObsId(int(obs_id.track_id), int(obs_id.frame))
        if obs_id not in self._res:
            raise UnknownObservationError(f"no residual for {obs_id}")
        return self._res[obs_id]

    def clear(self):
        self._res = {}

    def __contains__(self, obs_id):
        return ObsId(int(obs_id.track_id), int(obs_id.frame)) in self._res

    def __len__(self):
        return len(self._res)

    def items(self):
        return sorted(self._res.items())

    def stats(self):
        if not self._res:
            raise UnknownObservationError("residual layer is empty")
        arr = np.array([r for r, _ in self._res.values()])
        return {"mean_px": float(arr.mean()),
                "p95_px": float(np.percentile(arr, 95)),
                "n_outliers": int(sum(1 for _, o in self._res.values() if o))}


# --------------------------------------------------------------------------
# the graph: three layers + anchors + versioned, undoable edits
# --------------------------------------------------------------------------
class ObservationGraph:
    def __init__(self, n_frames):
        self.tracks = TrackStore()
        self.cameras = CameraStore()
        self.residuals = ResidualStore()
        self.anchors = AnchorSet()
        self.n_frames = int(n_frames)
        self.version = 0
        self._edits = []          # committed edit records, in order

    # -- construction ------------------------------------------------------
    @classmethod
    def from_tracks(cls, tracks, n_frames):
        g = cls(n_frames)
        for tr in tracks:
            g.tracks.add(TrackRecord(tr["id"], tr["frames"], tr["xy"]))
        return g

    def view(self):
        """Read-only snapshot for the solver (see sfm.solve_graph)."""
        return ObservationView(self)

    # -- editing -----------------------------------------------------------
    def begin_edit(self):
        return Edit(self, self.version)

    def commit_solution(self, solution):
        """Apply a solver result to the camera/residual layers as one
        transaction. This is the ONLY way solver output reaches the layers;
        the solver itself never sees the graph."""
        edit = self.begin_edit()
        edit.stage_solution(solution)
        return edit.commit()

    def _commit(self, edit):
        if edit.base_version != self.version:
            raise StaleEditError(
                f"edit opened at version {edit.base_version}, graph is at "
                f"{self.version}; re-open and retry")
        edit.apply()
        rec = {"id": len(self._edits), "base_version": edit.base_version,
               "version": self.version + 1, "ops": edit.serialized_ops(),
               "inverse": edit.serialized_inverse()}
        self._edits.append(rec)
        self.version += 1
        return rec["id"]

    def undo(self, edit_id):
        """Revert a committed edit. The undo is itself appended to the log so
        a reloaded project replays to the same state."""
        rec = self._edits[edit_id]
        if rec.get("undone"):
            raise ValueError(f"edit {edit_id} already undone")
        for inv in rec["inverse"]:
            self._apply_inverse(inv)
        rec["undone"] = True
        self._edits.append({"id": len(self._edits), "base_version": self.version,
                            "version": self.version + 1,
                            "ops": [], "inverse": [], "undo_of": edit_id})
        self.version += 1

    def _apply_inverse(self, inv):
        op = inv["op"]
        if op == "restore_track":
            self._restore_track(inv)
        elif op == "restore_anchor":
            self.anchors.upsert_distance_anchor(
                (inv["points"][0], inv["points"][1]), inv["meters"],
                anchor_id=inv["anchor_id"])
        elif op == "remove_anchor":
            self.anchors.remove_distance_anchor(inv["anchor_id"])
        elif op == "restore_ground":
            self.anchors.set_ground(inv["ground_ids"])
        elif op == "restore_solution_state":
            self.cameras = CameraStore()
            for f, (R, t) in inv["cameras"].items():
                self.cameras.set(int(f), R, t)
            self.residuals = ResidualStore()
            for tid, fr, r, outl in inv["residuals"]:
                self.residuals.set(ObsId(tid, fr), r, outl)
        else:
            raise CorruptEditLogError(f"unknown inverse op {op!r}")

    def _restore_track(self, inv):
        """Restore a deleted track, recovering the original constraint by
        OBSERVATION MEMBERS. The live fragments descending from the edited
        track (via their lineage) plus the observations this edit removed
        must union to exactly the edited track's member set; the edited id
        must not have been recycled onto an unrelated track. Anything else
        means the track lineage was modified after the edit - refuse and
        require re-identification instead of binding a same-name track."""
        tid = inv["track_id"]
        origin = inv["origin"]
        if tid in self.tracks and self.tracks.get(tid).origin != origin:
            raise ReidentifyRequired(
                f"track id {tid} is now occupied by an unrelated track "
                f"(origin {self.tracks.get(tid).origin}); re-identify members")
        want_all = {ObsId(origin, int(f)) for f in inv["frames"]}
        deleted = {ObsId(origin, int(f)) for f in inv["deleted_frames"]}
        expect_live = want_all - deleted
        frags = [self.tracks.get(i) for i in self.tracks.ids()
                 if tid in self.tracks.get(i).lineage]
        have = set()
        for fr in frags:
            have |= fr.members()
        if have != expect_live:
            raise ReidentifyRequired(
                f"track {origin} was split into {[f.id for f in frags]} and "
                f"its members no longer match the original observation set; "
                f"re-identify the fragments")
        for fr in frags:
            self.tracks.remove(fr.id)
        order = np.argsort(inv["frames"])
        frames = np.asarray(inv["frames"], np.int32)[order]
        xy = np.asarray(inv["xy"], np.float32)[order]
        self.tracks.add(TrackRecord(tid, frames, xy, origin=origin,
                                    lineage=inv["lineage"]))

    def edits(self):
        return list(self._edits)

    # -- persistence -------------------------------------------------------
    def save(self, run_dir):
        recs = self.tracks.snapshot()
        np.savez_compressed(
            os.path.join(run_dir, "obsgraph.npz"),
            n_frames=self.n_frames, version=self.version,
            track_id=np.array([t.id for t in recs], np.int32),
            track_origin=np.array([t.origin for t in recs], np.int32),
            track_lineage=np.array([json.dumps(list(t.lineage)) for t in recs]),
            frames=np.concatenate([t.frames for t in recs]).astype(np.int32)
            if recs else np.zeros(0, np.int32),
            xy=(np.concatenate([t.xy for t in recs]).astype(np.float32)
                if recs else np.zeros((0, 2), np.float32)),
            counts=np.array([len(t.frames) for t in recs], np.int32))
        with open(os.path.join(run_dir, "anchors.json"), "w") as fh:
            json.dump(self.anchors.to_dict(), fh)
        with open(os.path.join(run_dir, "edits.jsonl"), "w") as fh:
            for rec in self._edits:
                fh.write(json.dumps(rec) + "\n")

    @classmethod
    def load(cls, run_dir):
        z = np.load(os.path.join(run_dir, "obsgraph.npz"))
        g = cls(int(z["n_frames"]))
        counts = z["counts"]
        frames, xy = z["frames"], z["xy"]
        lineages = ([json.loads(s) for s in z["track_lineage"]]
                    if "track_lineage" in z else [[] for _ in counts])
        off = 0
        for k, tid in enumerate(z["track_id"]):
            n = int(counts[k])
            g.tracks.add(TrackRecord(int(tid), frames[off:off + n],
                                     xy[off:off + n], int(z["track_origin"][k]),
                                     lineage=lineages[k]))
            off += n
        ap = os.path.join(run_dir, "anchors.json")
        if os.path.exists(ap):
            with open(ap) as fh:
                g.anchors = AnchorSet.from_dict(json.load(fh))
        version = 0
        path = os.path.join(run_dir, "edits.jsonl")
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    if int(rec["base_version"]) != version:
                        raise CorruptEditLogError(
                            f"edit {rec.get('id')} commits on base "
                            f"{rec['base_version']} but the log is at {version}: "
                            f"two commits raced on one base and both were "
                            f"persisted - the project log is illegal")
                    version = int(rec["version"])
                    g._edits.append(rec)
        if int(z["version"]) != version:
            raise CorruptEditLogError(
                f"graph snapshot is at version {int(z['version'])} but the "
                f"edit log replays to {version}")
        g.version = version
        return g


class Edit:
    """One undoable transaction. Records its inverse as ops are staged;
    commit applies atomically under the version check."""

    def __init__(self, graph, base_version):
        self.graph = graph
        self.base_version = base_version
        self._ops = []
        self._inverse = []
        self._closed = False

    # -- staged ops --------------------------------------------------------
    def delete_observations(self, obs_ids):
        """Delete single observations by identity. Observations removed from
        the interior of a track split it: the fragments are re-registered
        under fresh ids (the old id is retired, never silently reused)."""
        by_track = {}
        for oid in obs_ids:
            oid = ObsId(int(oid.track_id), int(oid.frame))
            by_track.setdefault(oid.track_id, []).append(oid.frame)
        for tid, frames in by_track.items():
            rec = self.graph.tracks.get(tid)   # strict: unknown id raises
            self._ops.append({"op": "delete_observations", "track_id": tid,
                              "frames": [int(f) for f in frames]})
            self._inverse.append({
                "op": "restore_track", "track_id": tid, "origin": rec.origin,
                "frames": [int(f) for f in rec.frames],
                "xy": [list(map(float, p)) for p in rec.xy],
                "lineage": list(rec.lineage),
                "deleted_frames": [int(f) for f in frames]})
        return self

    def delete_track(self, tid):
        rec = self.graph.tracks.get(tid)
        return self.delete_observations(
            [ObsId(tid, int(f)) for f in rec.frames])

    def delete_scale_anchor(self, anchor_id):
        anc = self.graph.anchors.get_distance_anchor(anchor_id)  # strict
        self._ops.append({"op": "delete_scale_anchor", "anchor_id": int(anchor_id)})
        self._inverse.append({"op": "restore_anchor", "anchor_id": int(anchor_id),
                              "points": list(anc["points"]), "meters": anc["meters"]})
        return self

    def stage_anchors(self, dist_pair=None, dist_meters=None, ground_ids=()):
        if dist_pair is not None:
            points = (int(dist_pair[0]), int(dist_pair[1]))
            # resolve the target id now so the inverse is exact: replacing an
            # existing anchor restores its prior value; a new one is removed
            prior_id = None
            for aid, anc in self.graph.anchors.distance_anchors.items():
                if tuple(anc["points"]) == points:
                    prior_id = aid
                    break
            if prior_id is not None:
                prior = self.graph.anchors.get_distance_anchor(prior_id)
                self._inverse.append({"op": "restore_anchor",
                                      "anchor_id": int(prior_id),
                                      "points": list(prior["points"]),
                                      "meters": prior["meters"]})
            else:
                prior_id = self.graph.anchors.reserve_id()
                self._inverse.append({"op": "remove_anchor",
                                      "anchor_id": int(prior_id)})
            self._ops.append({"op": "upsert_distance_anchor",
                              "anchor_id": int(prior_id),
                              "points": list(points),
                              "meters": float(dist_meters)})
        ground_ids = [int(g) for g in ground_ids]
        if ground_ids:
            prior = sorted(self.graph.anchors.ground_ids)
            self._ops.append({"op": "add_ground", "ground_ids": ground_ids})
            self._inverse.append({"op": "restore_ground", "ground_ids": prior})
        return self

    def stage_solution(self, solution):
        cams = {int(f): (R.tolist(), t.tolist())
                for f, (R, t) in solution.cameras.items()}
        res = [[int(p), int(f), float(r), bool(o)]
               for (p, f, _, _), r, o in zip(solution.observations,
                                             solution.obs_residual,
                                             solution.obs_outlier)]
        prior_cams = {int(f): (R.tolist(), t.tolist())
                      for f, (R, t) in self.graph.cameras.as_dict().items()}
        prior_res = [[oid.track_id, oid.frame, r, o]
                     for oid, (r, o) in self.graph.residuals.items()]
        self._ops.append({"op": "apply_solution", "cameras": cams, "residuals": res})
        self._inverse.append({"op": "restore_solution_state",
                              "cameras": prior_cams, "residuals": prior_res})
        return self

    # -- transaction boundaries -------------------------------------------
    def serialized_ops(self):
        return self._ops

    def serialized_inverse(self):
        return self._inverse

    def apply(self):
        for op in self._ops:
            kind = op["op"]
            if kind == "delete_observations":
                self.graph.tracks.split(op["track_id"], op["frames"])
            elif kind == "delete_scale_anchor":
                self.graph.anchors.remove_distance_anchor(op["anchor_id"])
            elif kind == "upsert_distance_anchor":
                self.graph.anchors.upsert_distance_anchor(
                    (op["points"][0], op["points"][1]), op["meters"],
                    anchor_id=op["anchor_id"])
            elif kind == "add_ground":
                self.graph.anchors.add_ground(op["ground_ids"])
            elif kind == "apply_solution":
                for f, (R, t) in op["cameras"].items():
                    self.graph.cameras.set(int(f), R, t)
                self.graph.residuals.clear()
                for tid, fr, r, outl in op["residuals"]:
                    self.graph.residuals.set(ObsId(tid, fr), r, outl)
            else:
                raise CorruptEditLogError(f"unknown op {kind!r}")

    def commit(self):
        if self._closed:
            raise RuntimeError("edit already closed")
        self._closed = True
        return self.graph._commit(self)

    def rollback(self):
        self._closed = True
        self._ops = []
        self._inverse = []


class ObservationView:
    """Immutable snapshot of the track layer handed to the solver. The
    solver reads tracks through this view and can therefore never mutate
    the live graph (the UI-visible point set)."""

    def __init__(self, graph):
        self._tracks = [
            {"id": t.id, "frames": t.frames.copy(), "xy": t.xy.copy()}
            for t in graph.tracks.snapshot()]
        self.version = graph.version
        self.n_frames = graph.n_frames

    @classmethod
    def from_track_dicts(cls, tracks, n_frames):
        """Compatibility shim for the old solve(tracks, ...) entry."""
        v = cls.__new__(cls)
        v._tracks = [{"id": int(tr["id"]),
                      "frames": np.asarray(tr["frames"], np.int32).copy(),
                      "xy": np.asarray(tr["xy"], np.float32).reshape(-1, 2).copy()}
                     for tr in tracks]
        v.version = None
        v.n_frames = int(n_frames)
        return v

    def tracks(self):
        return list(self._tracks)
