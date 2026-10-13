"""Unit tests for RandomSplitter grouping."""

import numpy as np
import pytest

from modularml import FeatureSet


@pytest.fixture
def sorted_group_featureset():
    """20 groups of 5 rows, stored in sorted group order (as a loader would produce)."""
    groups = np.repeat([f"G{i:02d}" for i in range(20)], 5)
    return FeatureSet.from_dict(
        label="Grouped",
        data={
            "x": np.arange(len(groups), dtype=float),
            "y": np.arange(len(groups)) % 2,
            "group": groups,
        },
        feature_keys="x",
        target_keys="y",
        tag_keys="group",
    )


def _test_groups(fs, seed):
    views = fs.split_random(
        ratios={"train": 0.8, "test": 0.2},
        group_by="group",
        seed=seed,
        return_views=True,
        register=False,
    )
    groups = {
        k: set(np.asarray(v.get_tags(fmt="numpy", tags="group")).ravel().tolist())
        for k, v in views.items()
    }
    return groups["train"], groups["test"]


@pytest.mark.unit
def test_group_split_keeps_groups_whole(sorted_group_featureset):
    train, test = _test_groups(sorted_group_featureset, seed=0)
    assert train.isdisjoint(test)
    assert len(train) == 16
    assert len(test) == 4


@pytest.mark.unit
def test_group_split_depends_on_seed(sorted_group_featureset):
    """Group assignment is a seeded shuffle, not the order of first appearance."""
    test_sets = {
        frozenset(_test_groups(sorted_group_featureset, seed=s)[1]) for s in range(5)
    }
    assert len(test_sets) > 1
    last_four = frozenset(f"G{i:02d}" for i in range(16, 20))
    assert test_sets != {last_four}


@pytest.mark.unit
def test_group_split_is_reproducible(sorted_group_featureset):
    assert _test_groups(sorted_group_featureset, seed=7) == _test_groups(
        sorted_group_featureset,
        seed=7,
    )
