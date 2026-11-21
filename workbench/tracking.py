"""Cross-frame feature tracking with pyramidal Lucas-Kanade and
forward-backward consistency, plus synthetic wrong-match injection."""
import os

import cv2
import numpy as np


def _detect(gray, max_corners, quality, min_distance, mask=None):
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality,
                                  minDistance=min_distance, blockSize=7, mask=mask)
    if pts is None:
        return np.zeros((0, 2), np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def _flow(prev, nxt, pts, guess=None):
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    pts_r = pts.reshape(-1, 1, 2)
    if guess is not None:
        init = guess.reshape(-1, 1, 2).astype(np.float32)
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    else:
        init = None
        flags = 0
    nxt_pts, st, _ = cv2.calcOpticalFlowPyrLK(
        prev, nxt, pts_r, init, winSize=(15, 15), maxLevel=3,
        criteria=crit, flags=flags)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(
        nxt, prev, nxt_pts, None, winSize=(15, 15), maxLevel=3,
        criteria=crit)
    fb_err = np.linalg.norm(pts_r - back, axis=2).reshape(-1)
    ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb_err < 0.7)
    return nxt_pts.reshape(-1, 2), ok


def build_tracks(frames_dir, max_tracks=600, min_distance=10, quality=1e-3,
                 max_track_len=34, seed=0):
    """Track features through the sequence. Tracks are retired after a
    randomized lifetime (~max_track_len, jittered to avoid synchronized death
    of the whole map) to bound KLT drift; features then respawn as new tracks.
    The lifetime must comfortably exceed the longest pure-rotation segment, or
    no landmark bridges the segment and the two translation sides drift apart
    in scale. Returns (tracks, n_frames)."""
    rng = np.random.default_rng(seed)
    life = {}   # per-track retirement age

    def track_life(tid):
        if tid not in life:
            life[tid] = int(max_track_len + rng.integers(-4, 5))
        return life[tid]
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    n_frames = len(files)
    gray0 = cv2.imread(os.path.join(frames_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    h, w = gray0.shape

    tracks = []
    active = {}   # track_id -> current xy
    prev_pts_ids = []
    prev = gray0
    next_id = 0

    def spawn(gray, frame_idx, existing_xy):
        nonlocal next_id
        mask = np.full((h, w), 255, np.uint8)
        for (x, y) in existing_xy:
            cv2.circle(mask, (int(round(x)), int(round(y))), min_distance, 0, -1)
        want = max_tracks - len(existing_xy)
        if want <= 0:
            return
        pts = _detect(gray, want, quality, min_distance, mask)
        for p in pts:
            tracks.append({"id": next_id, "start": frame_idx,
                           "frames": [frame_idx], "xy": [p.copy()]})
            active[next_id] = p
            prev_pts_ids.append(next_id)
            next_id += 1

    spawn(prev, 0, [])
    for fi in range(1, n_frames):
        gray = cv2.imread(os.path.join(frames_dir, files[fi]), cv2.IMREAD_GRAYSCALE)
        if prev_pts_ids:
            pts = np.array([active[tid] for tid in prev_pts_ids], np.float32)
            guess = []
            for tid in prev_pts_ids:
                tr = tracks[tid]
                if len(tr["xy"]) >= 2:
                    guess.append(2 * tr["xy"][-1] - tr["xy"][-2])
                else:
                    guess.append(tr["xy"][-1])
            nxt, ok = _flow(prev, gray, pts, np.asarray(guess, np.float32))
            still_ids, still_pts = [], []
            for tid, p, good in zip(prev_pts_ids, nxt, ok):
                x, y = p
                tr = tracks[tid]
                if good and 0 <= x < w and 0 <= y < h and len(tr["frames"]) < track_life(tid):
                    active[tid] = p
                    tr["frames"].append(fi)
                    tr["xy"].append(p.copy())
                    still_ids.append(tid)
                    still_pts.append(p)
                else:
                    active.pop(tid, None)   # lost or retired (drift bound)
            prev_pts_ids = still_ids
        if len(prev_pts_ids) < int(0.75 * max_tracks):
            spawn(gray, fi, [active[t] for t in prev_pts_ids])
        prev = gray

    for tr in tracks:
        tr["xy"] = np.asarray(tr["xy"], np.float32).reshape(-1, 2)
        tr["frames"] = np.asarray(tr["frames"], np.int32)
    return tracks, n_frames


def inject_outliers(tracks, ratio, seed=0, magnitude=(15.0, 60.0)):
    """Corrupt a fraction of observations to simulate wrong matches.
    Returns a boolean mask list aligned with observations: set of (track_id, obs_idx)."""
    rng = np.random.default_rng(seed)
    injected = []
    for tr in tracks:
        m = len(tr["frames"])
        if m < 3:
            continue
        k = rng.random(m) < ratio
        # never corrupt the first observation (keeps track birth honest)
        k[0] = False
        idxs = np.nonzero(k)[0]
        for i in idxs:
            ang = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(*magnitude)
            tr["xy"][i] += mag * np.array([np.cos(ang), np.sin(ang)], np.float32)
            injected.append((tr["id"], int(i)))
    return injected


def tracks_to_arrays(tracks):
    """Flatten tracks into observation arrays."""
    obs_track, obs_frame, obs_xy = [], [], []
    for tr in tracks:
        for j, f in enumerate(tr["frames"]):
            obs_track.append(tr["id"])
            obs_frame.append(int(f))
            obs_xy.append(tr["xy"][j])
    return (np.asarray(obs_track, np.int32), np.asarray(obs_frame, np.int32),
            np.asarray(obs_xy, np.float32).reshape(-1, 2))
