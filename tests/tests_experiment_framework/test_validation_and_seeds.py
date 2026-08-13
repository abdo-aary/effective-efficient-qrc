from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.experiment.domain import (
    AcquisitionKind,
    AcquisitionSpec,
    PrefixAxis,
    ResourceLimit,
)
from src.experiment.seeding import PathSeedTree

from .helpers import minimal_plan


def test_path_seed_streams_are_order_independent_and_distinct():
    first = PathSeedTree.from_root(1234)
    train_before = first.generator("data/train").normal(size=8)
    first.generator("unrelated/new-stream").normal(size=3)
    train_after = first.generator("data/train").normal(size=8)
    second = PathSeedTree.from_root(1234)
    np.testing.assert_array_equal(train_before, train_after)
    np.testing.assert_array_equal(train_before, second.generator("data/train").normal(size=8))
    assert first.integer("measurement/train") != first.integer("measurement/heldout")
    full = first.generator("design/candidate=7").normal(size=16)
    prefix = first.generator("design/candidate=7").normal(size=4)
    np.testing.assert_array_equal(prefix, full[:4])


def test_cartesian_r_s_m_prefix_cache_is_rejected():
    with pytest.raises(ValueError, match="R, S, and M"):
        AcquisitionSpec(
            id="bad",
            data_id="data/train",
            kind=AcquisitionKind.GROUPED_MEASUREMENT,
            split="train",
            prefix_limits=(
                ResourceLimit(PrefixAxis.R, 32),
                ResourceLimit(PrefixAxis.S, 256),
                ResourceLimit(PrefixAxis.M, 8192),
            ),
            fixed_resources={},
            randomness_path="measurement/train",
            pairing_key="pair",
            program_key="quark",
        )


def test_invalid_prefix_and_test_data_fit_are_rejected():
    plan = minimal_plan()
    train_view = plan.feature_views[0]
    invalid_view = replace(
        train_view,
        prefixes=(ResourceLimit(PrefixAxis.R, 2),),
    )
    with pytest.raises(ValueError, match="outside acquisition"):
        replace(plan, feature_views=(invalid_view, plan.feature_views[1]))

    invalid_fit = replace(plan.fits[0], feature_view_id="view/test")
    with pytest.raises(ValueError, match="non-training data"):
        replace(plan, fits=(invalid_fit,))


def test_paired_comparison_denominator_mismatch_is_rejected():
    plan = minimal_plan()
    bad = replace(plan.comparisons[0], denominator_key="other-labels")
    with pytest.raises(ValueError, match="one denominator"):
        replace(plan, comparisons=(bad,))



def test_mismatched_evaluation_pairing_is_rejected():
    plan = minimal_plan()
    bad_evaluation = replace(plan.evaluations[0], pairing_key="another/pair")

    with pytest.raises(ValueError, match="experiment pairing"):
        replace(plan, evaluations=(bad_evaluation,))
