"""Readout-boundary feature flattening."""

from __future__ import annotations

from src.core.results import FeatureBatch


def flatten_for_readout(batch: FeatureBatch):
    return batch.flatten_for_readout()
