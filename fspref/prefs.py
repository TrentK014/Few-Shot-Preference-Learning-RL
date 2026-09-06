"""Segment sampling, automatic preference labeling, and the on-disk dataset.

This is the module every later milestone imports. A preference dataset is one
(family, goal variation) pair's worth of transitions plus a table of labeled
segment pairs referencing them by index. Segments are never materialized on disk.

Labels come from MetaWorld's ground-truth reward, which acts as a synthetic
human: the segment with the higher return is preferred. The numeric returns are
stored for diagnostics only and must never be used as regression targets.
"""
from __future__ import annotations

import os

import numpy as np

from .collect import SOURCES

SEGMENT_SIZE = 25  # paper Table 1, MetaWorld
N_PAIRS = 6000     # reference dataset_kwargs.capacity


def valid_segment_starts(episode_id: np.ndarray, segment_size: int = SEGMENT_SIZE) -> np.ndarray:
    """Start indices whose whole `segment_size` window stays inside one episode.

    Equivalent to the reference's rejection of any window containing a `done`.
    """
    n = len(episode_id)
    if n < segment_size:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(n - segment_size + 1, dtype=np.int64)
    # A window is valid iff its first and last step belong to the same episode.
    same = episode_id[starts] == episode_id[starts + segment_size - 1]
    return starts[same]


def segment_returns(reward: np.ndarray, starts: np.ndarray, segment_size: int = SEGMENT_SIZE,
                    discount: float = 1.0) -> np.ndarray:
    """Sum of ground-truth reward over each segment.

    discount=1.0 is the paper's plain sum. The reference code instead uses
    gamma=0.99 over the 25 steps; pass discount=0.99 to match it exactly.
    """
    idx = starts[:, None] + np.arange(segment_size)[None, :]
    seg = reward[idx]
    if discount != 1.0:
        seg = seg * (discount ** np.arange(segment_size))[None, :]
    return seg.sum(axis=1)


def build_pairs(data: dict, n_pairs: int = N_PAIRS, segment_size: int = SEGMENT_SIZE,
                discount: float = 1.0, tie_margin: float = 0.0, seed: int = 0,
                candidate_starts: np.ndarray | None = None, oversample: int = 8,
                max_resample_rounds: int = 50) -> dict:
    """Sample `n_pairs` segment pairs and label them by ground-truth return.

    Pairs are drawn only from within this variation's data. Label 1 means the
    first segment is preferred.

    candidate_starts restricts the pool to a given set of segment starts, which
    is how held-out-episode evaluation pairs and Milestone 8's query budgets are
    drawn. None means every valid segment in the variation.

    tie_margin drops pairs whose ground-truth returns differ by less than the
    margin, and oversamples so the dataset still reaches n_pairs. A pair the
    oracle cannot separate carries no information and its label is arbitrary.
    """
    rng = np.random.RandomState(seed)
    starts = (valid_segment_starts(data["episode_id"], segment_size)
              if candidate_starts is None else np.asarray(candidate_starts, dtype=np.int64))
    if len(starts) == 0:
        raise ValueError("no valid segments: episodes shorter than the segment size")

    if tie_margin <= 0:
        a = starts[rng.randint(len(starts), size=n_pairs)]
        b = starts[rng.randint(len(starts), size=n_pairs)]
        ret_a = segment_returns(data["reward"], a, segment_size, discount)
        ret_b = segment_returns(data["reward"], b, segment_size, discount)
    else:
        # Oversample and filter, so a margin does not silently shrink the dataset.
        # Drawer Close needs this badly: its reward is effectively binary (zero
        # until the drawer shuts, then 10), so 74% of uniformly sampled pairs
        # differ by less than 1e-6 and their labels are coin flips. Window Open's
        # dense reward leaves under 4% such pairs.
        keep_a, keep_b, keep_ra, keep_rb = [], [], [], []
        kept = 0
        for _ in range(max_resample_rounds):
            draw = max(n_pairs - kept, 1) * oversample
            a = starts[rng.randint(len(starts), size=draw)]
            b = starts[rng.randint(len(starts), size=draw)]
            ra = segment_returns(data["reward"], a, segment_size, discount)
            rb = segment_returns(data["reward"], b, segment_size, discount)
            m = np.abs(ra - rb) >= tie_margin
            keep_a.append(a[m]); keep_b.append(b[m])
            keep_ra.append(ra[m]); keep_rb.append(rb[m])
            kept += int(m.sum())
            if kept >= n_pairs:
                break
        a = np.concatenate(keep_a)[:n_pairs]
        b = np.concatenate(keep_b)[:n_pairs]
        ret_a = np.concatenate(keep_ra)[:n_pairs]
        ret_b = np.concatenate(keep_rb)[:n_pairs]
    return dict(pair_a_start=a.astype(np.int64), pair_b_start=b.astype(np.int64),
                label=(ret_a > ret_b).astype(np.float32),
                ret_a=ret_a.astype(np.float32), ret_b=ret_b.astype(np.float32))


def save_dataset(path: str, data: dict, pairs: dict, meta: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {**data, **pairs}
    payload.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    np.savez_compressed(path, **payload)
    return path


def load_dataset(path: str) -> dict:
    """Load one variation's dataset. Returns arrays plus a `meta` dict."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if not k.startswith("meta_")}
    out["meta"] = {k[len("meta_"):]: z[k].item() if z[k].ndim == 0 else z[k]
                   for k in z.files if k.startswith("meta_")}
    return out


def materialize(data: dict, starts: np.ndarray, segment_size: int = SEGMENT_SIZE,
                mask_goal: bool = False):
    """Gather (N, segment_size, 39) observations and (N, segment_size, 4) actions.

    Observations are already goal-masked at collection time; mask_goal is a
    belt-and-braces option for data collected goal-visible.
    """
    idx = starts[:, None] + np.arange(segment_size)[None, :]
    obs = data["obs"][idx]
    act = data["action"][idx]
    if mask_goal:
        obs = obs.copy()
        obs[..., 36:39] = 0.0
    return obs, act


def segment_source(data: dict, starts: np.ndarray) -> np.ndarray:
    """Behavior source id of each segment (segments never cross episodes)."""
    return data["source"][data["episode_id"][starts]]


def variation_path(root: str, family: str, variation: int) -> str:
    return os.path.join(root, family, f"var{variation:02d}.npz")


def list_datasets(root: str, family: str | None = None):
    """All dataset paths under `root`, optionally for one family, sorted."""
    families = [family] if family else sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    paths = []
    for fam in families:
        d = os.path.join(root, fam)
        if os.path.isdir(d):
            paths += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".npz")]
    return paths


def task_key(path: str) -> str:
    """MAML task key: family plus goal variation, matching the reference's per-variation keying."""
    return f"{os.path.basename(os.path.dirname(path))}/{os.path.splitext(os.path.basename(path))[0]}"


__all__ = ["SEGMENT_SIZE", "N_PAIRS", "SOURCES", "valid_segment_starts", "segment_returns",
           "build_pairs", "save_dataset", "load_dataset", "materialize", "segment_source",
           "variation_path", "list_datasets", "task_key"]
