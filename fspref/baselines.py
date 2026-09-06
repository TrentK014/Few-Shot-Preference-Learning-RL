"""Heuristic baselines that calibrate what a preference accuracy number means.

MetaWorld's dense reward is dominated by its reaching term, so "whichever segment
kept the hand closer to the object" already predicts about 92% of preference
labels on Window Open. Any learned reward model must be measured against that,
not against chance, or an 85% result reads as success while being worse than one
line of arithmetic.

The hard subset is the other half of the story: restricted to pairs whose two
segments have nearly equal hand-to-object distance, the shortcut collapses to
about 72%. That subset is where task understanding actually shows up.

Nothing here is hardcoded; every bar is recomputed from whatever data is passed.
"""
from __future__ import annotations

import numpy as np

HAND_SLICE = slice(0, 3)
OBJ_SLICE = slice(4, 7)
PREV_OBJ_SLICE = slice(22, 25)  # obs[18:36] is the previous frame, so 18+4 .. 18+7


def _segment_view(obs: np.ndarray, starts: np.ndarray, segment_size: int) -> np.ndarray:
    idx = starts[:, None] + np.arange(segment_size)[None, :]
    return obs[idx]


def hand_object_distance(obs: np.ndarray, starts: np.ndarray, segment_size: int) -> np.ndarray:
    """Negative mean hand-to-object distance over each segment (higher is 'better')."""
    seg = _segment_view(obs, starts, segment_size)
    return -np.linalg.norm(seg[..., HAND_SLICE] - seg[..., OBJ_SLICE], axis=-1).mean(axis=1)


def object_displacement_x(obs: np.ndarray, starts: np.ndarray, segment_size: int) -> np.ndarray:
    seg = _segment_view(obs, starts, segment_size)
    return seg[:, -1, OBJ_SLICE.start] - seg[:, 0, OBJ_SLICE.start]


def object_velocity_x(obs: np.ndarray, starts: np.ndarray, segment_size: int) -> np.ndarray:
    seg = _segment_view(obs, starts, segment_size)
    return (seg[..., OBJ_SLICE.start] - seg[..., PREV_OBJ_SLICE.start]).mean(axis=1)


def object_position_x(obs: np.ndarray, starts: np.ndarray, segment_size: int) -> np.ndarray:
    return _segment_view(obs, starts, segment_size)[..., OBJ_SLICE.start].mean(axis=1)


FEATURES = {
    "hand_obj_distance": hand_object_distance,
    "obj_displacement_x": object_displacement_x,
    "obj_velocity_x": object_velocity_x,
    "obj_position_x": object_position_x,
}


def feature_scores(obs: np.ndarray, a_starts: np.ndarray, b_starts: np.ndarray,
                   segment_size: int, name: str = "hand_obj_distance"):
    """Heuristic score of each side of every pair."""
    f = FEATURES[name]
    return f(obs, a_starts, segment_size), f(obs, b_starts, segment_size)


def heuristic_accuracy(obs: np.ndarray, a_starts: np.ndarray, b_starts: np.ndarray,
                       label: np.ndarray, segment_size: int,
                       name: str = "hand_obj_distance") -> float:
    fa, fb = feature_scores(obs, a_starts, b_starts, segment_size, name)
    return float(((fa > fb).astype(np.float32) == label).mean())


def best_single_feature(obs: np.ndarray, a_starts: np.ndarray, b_starts: np.ndarray,
                        label: np.ndarray, segment_size: int):
    """(name, accuracy) of the strongest hand-picked feature. The bar to beat."""
    scored = [(heuristic_accuracy(obs, a_starts, b_starts, label, segment_size, n), n)
              for n in FEATURES]
    acc, name = max(scored)
    return name, acc


def hard_subset_mask(obs: np.ndarray, a_starts: np.ndarray, b_starts: np.ndarray,
                     segment_size: int, quantile: float = 0.25,
                     name: str = "hand_obj_distance") -> np.ndarray:
    """Pairs where the shortcut feature is least informative: the smallest gaps.

    These are the pairs that require understanding the task rather than noticing
    that one arm was closer to the object.
    """
    fa, fb = feature_scores(obs, a_starts, b_starts, segment_size, name)
    gap = np.abs(fa - fb)
    return gap <= np.quantile(gap, quantile)


def calibration_report(obs: np.ndarray, a_starts: np.ndarray, b_starts: np.ndarray,
                       label: np.ndarray, segment_size: int, quantile: float = 0.25) -> dict:
    """Every bar a learned model has to clear on this exact set of pairs."""
    name, best = best_single_feature(obs, a_starts, b_starts, label, segment_size)
    hard = hard_subset_mask(obs, a_starts, b_starts, segment_size, quantile)
    hard_acc = heuristic_accuracy(obs, a_starts[hard], b_starts[hard], label[hard], segment_size)
    return dict(best_feature=name, best_feature_acc=best,
                hand_dist_acc=heuristic_accuracy(obs, a_starts, b_starts, label, segment_size),
                hard_mask=hard, hard_heuristic_acc=hard_acc, n_hard=int(hard.sum()))
