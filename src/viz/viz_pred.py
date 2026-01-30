import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from typing import Literal, Sequence, Optional, Union, Tuple

from src.experiment.experiment import Experiment


def _as_KN(y: np.ndarray) -> np.ndarray:
    """Canonicalize labels/preds to shape (K, N)."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y[None, :]
    if y.ndim != 2:
        raise ValueError(f"y must be 1D/2D, got {y.shape}")
    # heuristic: N usually larger than K
    return y.T if y.shape[0] > y.shape[1] else y


def _get_task_names(exp) -> list[str]:
    """Best-effort task names."""
    ds = exp.dataset
    if getattr(ds, "label_functionals", None):
        return [type(f).__name__ for f in ds.label_functionals]
    # fallback
    y = _as_KN(ds.y)
    return [f"y{k}" for k in range(y.shape[0])]


def plot_true_vs_pred_grid(
    exp,
    task_indices=None,
    *,
    split="test",
    x_axis="relative",
    max_points=300,
    sharex=True,
    sharey=False,
    figsize: Optional[Tuple[float, float]] = None,
    panel_size: Tuple[float, float] = (5.0, 3.2),
) -> plt.Figure:
    """
    Multi-panel (squarish) plot: one axis per task showing y_true vs y_pred.

    Parameters
    ----------
    exp:
        Experiment instance with exp.dataset loaded and exp.model loaded.
        The model must have:
          - Phi_full_
          - train_idx_ / test_idx_
          - predict_from_features(...)
    task_indices:
        None -> plot all tasks
        int  -> plot that task only
        seq  -> plot selected tasks
    split:
        "test" (default) or "train"
    x_axis:
        "relative" -> x = 0..n-1 (after thinning)
        "original" -> x = window indices in the dataset
    max_points:
        If not None, thin the chosen split indices to at most max_points (evenly spaced).
    sharex:
        Passed to matplotlib subplots for nicer alignment.
    sharey:
        Passed to matplotlib subplots for nicer alignment.

    Returns
    -------
    fig : matplotlib.figure.Figure
    :param sharex:
    """
    model = exp.model
    ds = exp.dataset

    y = _as_KN(ds.y)  # (K, N)
    K, N = y.shape

    idx = np.asarray(model.test_idx_ if split == "test" else model.train_idx_, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError(f"Empty or invalid {split}_idx_")

    # thin for readability
    if max_points is not None and idx.size > max_points:
        keep = np.linspace(0, idx.size - 1, num=max_points).round().astype(int)
        idx = idx[keep]

    # resolve task list
    if task_indices is None:
        tasks = list(range(K))
    elif isinstance(task_indices, int):
        tasks = [int(task_indices)]
    else:
        tasks = [int(k) for k in task_indices]

    # validate & unique preserve order
    seen = set()
    tasks_clean = []
    for k in tasks:
        if k in seen:
            continue
        if not (0 <= k < K):
            raise ValueError(f"task index {k} out of range [0,{K-1}]")
        tasks_clean.append(k)
        seen.add(k)
    tasks = tasks_clean

    # get predictions on those indices using cached features
    if model.Phi_full_ is None:
        raise RuntimeError("model.Phi_full_ is None; did you load/fit the model?")
    Phi = model.Phi_full_[idx]
    y_pred = _as_KN(model.predict_from_features(Phi, apply_scaler=True))  # (K, nsub)

    # true labels
    y_true = y[:, idx]  # (K, nsub)

    # x axis
    x = np.arange(idx.size) if x_axis == "relative" else idx

    # layout: squarish grid
    nplots = len(tasks)
    ncols = int(ceil(sqrt(nplots)))
    nrows = int(ceil(nplots / ncols))

    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
    )
    axes = np.atleast_1d(axes).ravel()

    names = _get_task_names(exp)

    for i, k in enumerate(tasks):
        ax = axes[i]
        mse = float(np.mean((y_true[k] - y_pred[k]) ** 2))

        ax.plot(x, y_true[k], linestyle="-", label="true")
        ax.plot(x, y_pred[k], linestyle="--", label="pred")

        ax.set_title(f"{names[k]} ({split})  MSE={mse:.3g}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

        if i % ncols == 0:
            ax.set_ylabel("y")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("index" if x_axis == "relative" else "window index")

    # turn off unused axes
    for j in range(nplots, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig
