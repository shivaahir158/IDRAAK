"""Reproducible train/validation/test splitting."""

from __future__ import annotations

import random
from typing import Any

from idraak.schemas.dataset import DatasetEntry


def create_splits(
    entries: list[DatasetEntry],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    group_by_base_id: bool = True,
) -> dict[str, list[DatasetEntry]]:
    """Split entries into train/val/test, keeping grouped requirements together.

    All perturbations derived from one base requirement stay in the same split.
    """
    rng = random.Random(seed)

    if group_by_base_id:
        # Group by base requirement ID (strip perturbation suffixes)
        groups: dict[str, list[DatasetEntry]] = {}
        for e in entries:
            base_id = e.requirement_id.split("-PERT")[0].split("-PARA")[0]
            groups.setdefault(base_id, []).append(e)
        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)

        n = len(group_keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_keys = group_keys[:n_train]
        val_keys = group_keys[n_train : n_train + n_val]
        test_keys = group_keys[n_train + n_val :]

        return {
            "train": [e for k in train_keys for e in groups[k]],
            "validation": [e for k in val_keys for e in groups[k]],
            "test": [e for k in test_keys for e in groups[k]],
        }
    else:
        shuffled = list(entries)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return {
            "train": shuffled[:n_train],
            "validation": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
