"""Torch-side loading and splitting of preference datasets.

One variation holds only ~5,000 transitions (about 900 KB as float32), so the raw
observation and action arrays live on the device and segments are gathered by
index at batch time. Materializing 6,000 pairs of 25-step segments would cost
about 50 MB per variation for no benefit.

Three splits, because the obvious one is misleading. The 6,000 stored pairs of a
variation touch 96% of its ~3,900 distinct segments, so a held-out *pair* almost
always consists of two segments the model already saw inside other training
pairs. That measures memorization of segment scores, not generalization. The
held-out *episode* split is the honest one.
"""
from __future__ import annotations

import numpy as np
import torch

from .collect import SOURCES
from .prefs import (SEGMENT_SIZE, build_pairs, load_dataset, task_key,
                    valid_segment_starts)


def stratified_episode_holdout(data: dict, frac: float = 0.2, seed: int = 0):
    """Hold out `frac` of each behavior source's episodes, split into val and test.

    Stratification matters twice over, and getting only the first half right
    produced a real bug. A plain random holdout of 10 of 52 episodes drew
    3 expert / 6 within / 1 cross / 0 random. Worse, stratifying *which* episodes
    are held out but then splitting them into val and test at random gave
    val = 2 expert / 2 within / 1 cross and test = 1 expert / 3 within / 1 cross
    / 1 random. Expert-versus-random pairs are easy and within-versus-within
    pairs are hard, so validation accuracy read 0.966 while test read 0.880 on
    what should be the same distribution, and early stopping optimized the wrong
    thing.

    So episodes of each source are dealt alternately into val and test, keeping
    both groups the same mixture.

    Returns (held, val_episodes, test_episodes).
    """
    rng = np.random.RandomState(seed)
    src = data["source"]
    val, test = [], []
    flip = 0  # runs across sources so an odd count does not always favour val
    for i in range(len(SOURCES)):
        idx = np.where(src == i)[0]
        if len(idx) == 0:
            continue
        # At least 2 where possible so both groups can be represented, but never
        # take every episode of a source away from training.
        k = min(max(2, int(round(frac * len(idx)))), max(1, len(idx) - 1))
        chosen = rng.permutation(idx)[:k]
        for j, ep in enumerate(chosen):
            (val if (j + flip) % 2 == 0 else test).append(int(ep))
        flip += k
    val = np.array(sorted(val), dtype=np.int64)
    test = np.array(sorted(test), dtype=np.int64)
    held = np.array(sorted(val.tolist() + test.tolist()), dtype=np.int64)
    return held, val, test


class VariationData:
    """One (family, variation): transitions on device plus index-based pair tables."""

    def __init__(self, path: str, device="cpu", segment_size: int = SEGMENT_SIZE,
                 holdout_frac: float = 0.2, seed: int = 0):
        self.path = path
        self.key = task_key(path)
        raw = load_dataset(path)
        self.meta = raw["meta"]
        self.segment_size = segment_size
        self.device = device

        self.obs_np = raw["obs"]
        self.action_np = raw["action"]
        self.reward_np = raw["reward"]
        self.episode_id = raw["episode_id"]
        self.source = raw["source"]
        self.obs = torch.as_tensor(raw["obs"], dtype=torch.float32, device=device)
        self.action = torch.as_tensor(raw["action"], dtype=torch.float32, device=device)

        self.all_starts = valid_segment_starts(self.episode_id, segment_size)
        # The reference has no validation split at all and stops on training
        # accuracy, which underfits badly offline (0.95 train accuracy is reached
        # in 8 epochs while the honest metric is still climbing). Early stopping
        # needs a signal that is neither the training set nor the test set, and
        # both groups must have the same source mixture.
        (self.held_episodes, self.val_episodes,
         self.test_episodes) = stratified_episode_holdout(raw, holdout_frac, seed)
        ep_of_start = self.episode_id[self.all_starts]
        in_held = np.isin(ep_of_start, self.held_episodes)
        self.train_starts = self.all_starts[~in_held]
        self.held_starts = self.all_starts[in_held]
        self.val_starts = self.all_starts[np.isin(ep_of_start, self.val_episodes)]
        self.test_starts = self.all_starts[np.isin(ep_of_start, self.test_episodes)]

        # Split A: the stored pair table, shuffled into train/val/test.
        n = len(raw["label"])
        order = np.random.RandomState(seed).permutation(n)
        n_tr, n_va = int(0.8 * n), int(0.1 * n)
        self._stored = dict(a=raw["pair_a_start"], b=raw["pair_b_start"], y=raw["label"])
        self.pair_splits = dict(train=order[:n_tr], val=order[n_tr:n_tr + n_va],
                                test=order[n_tr + n_va:])
        # Training pairs must not touch held-out episodes, or the honest split leaks.
        held_set = set(self.held_starts.tolist())
        keep = np.array([a not in held_set and b not in held_set
                         for a, b in zip(self._stored["a"][self.pair_splits["train"]],
                                         self._stored["b"][self.pair_splits["train"]])])
        self.pair_splits["train"] = self.pair_splits["train"][keep]
        self.raw = raw

    def stored_pairs(self, split: str):
        idx = self.pair_splits[split]
        return (self._stored["a"][idx], self._stored["b"][idx], self._stored["y"][idx])

    def episode_holdout_pairs(self, n_pairs: int = 2000, seed: int = 0,
                              discount: float | None = None, which: str = "test"):
        """Fresh pairs drawn only from held-out episodes: the honest evaluation.

        which='val' feeds early stopping, which='test' is the reported number.
        The two draw from disjoint sets of episodes.
        """
        pool = {"val": self.val_starts, "test": self.test_starts,
                "all": self.held_starts}[which]
        disc = float(self.meta["discount"]) if discount is None else discount
        p = build_pairs(self.raw, n_pairs=n_pairs, segment_size=self.segment_size,
                        discount=disc, seed=seed, candidate_starts=pool)
        return p["pair_a_start"], p["pair_b_start"], p["label"]

    def gather(self, starts: np.ndarray):
        """(N, T, 39) observations and (N, T, 4) actions for these segment starts."""
        idx = torch.as_tensor(starts[:, None] + np.arange(self.segment_size)[None, :],
                              dtype=torch.long, device=self.device)
        return self.obs[idx], self.action[idx]

    def batch(self, a_starts, b_starts, labels):
        obs_a, act_a = self.gather(a_starts)
        obs_b, act_b = self.gather(b_starts)
        y = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        return obs_a, act_a, obs_b, act_b, y


class PreferenceData:
    """A collection of variations, iterated as shuffled minibatches of pairs."""

    def __init__(self, paths, device="cpu", segment_size: int = SEGMENT_SIZE,
                 holdout_frac: float = 0.2, seed: int = 0):
        self.variations = [VariationData(p, device, segment_size, holdout_frac, seed)
                           for p in paths]
        self.device = device

    def __len__(self):
        return len(self.variations)

    def iter_batches(self, split: str = "train", batch_size: int = 64, seed: int = 0,
                     shuffle: bool = True):
        """One pass over every variation's pairs of this split, in mixed batches.

        Batches never span variations, since segment indices are variation-local.
        """
        rng = np.random.RandomState(seed)
        order = []
        for vi, v in enumerate(self.variations):
            a, b, y = v.stored_pairs(split)
            order += [(vi, i) for i in range(len(y))]
        if shuffle:
            rng.shuffle(order)
        by_var: dict[int, list[int]] = {}
        for vi, i in order:
            by_var.setdefault(vi, []).append(i)
        # Emit batch-sized chunks per variation, interleaved in shuffled order.
        chunks = []
        for vi, idxs in by_var.items():
            a, b, y = self.variations[vi].stored_pairs(split)
            for s in range(0, len(idxs), batch_size):
                sel = np.array(idxs[s:s + batch_size])
                chunks.append((vi, a[sel], b[sel], y[sel]))
        if shuffle:
            rng.shuffle(chunks)
        for vi, a, b, y in chunks:
            yield self.variations[vi].batch(a, b, y)
