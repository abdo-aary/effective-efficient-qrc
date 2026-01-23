# from __future__ import annotations
#
# from typing import Any, Dict
#
# from .generate.varma import VARMAGenerator
# from .label.base import LabelNoise
# from .label.functionals import (BaseLabelFunctional, OneStepForecastFunctional,
#                                 ExpFadingLinearFunctional, VolteraFunctional)
#
#
# def build_generator(cfg: Dict[str, Any]):
#     """Build a window generator from a YAML-parsed dict.
#
#     """
#     ds = cfg.get("dataset", {})
#     proc = cfg.get("process", {})
#
#     kind = str(ds.get("generator", "arma")).lower()
#     if kind == "varma":
#         # return VARMAGenerator()
#         raise NotImplementedError()
#     else:
#         raise ValueError(f"Unknown generator kind={kind!r}.")
#
#
# def build_label_functional(cfg: Dict[str, Any]):
#     """Build a label functional from a YAML-parsed dict.
#
#     Expected structure:
#       label:
#         kind: exp_fading_linear | last_value | quadratic
#         noise_std: 0.0
#         ... kind-specific params ...
#     """
#     lab = cfg.get("label", {})
#     kind = str(lab.get("kind", "exp_fading_linear")).lower()
#     noise = LabelNoise(std=float(lab.get("noise_std", 0.0)))
#
#     if kind == "last_value":
#         return LastValueLabel(target_dim=int(lab.get("target_dim", 0)), noise=noise)
#
#     if kind == "exp_fading_linear":
#         return ExpFadingLinearLabel(
#             gamma=float(lab.get("gamma", 0.9)),
#             weight_scale=float(lab.get("weight_scale", 1.0)),
#             nonlinearity=str(lab.get("nonlinearity", "none")),
#             seed_weights=int(lab.get("seed_weights", 0)),
#             noise=noise,
#         )
#
#     if kind == "quadratic":
#         return QuadraticInteractionLabel(
#             a_scale=float(lab.get("a_scale", 1.0)),
#             b_scale=float(lab.get("b_scale", 0.25)),
#             seed_weights=int(lab.get("seed_weights", 0)),
#             noise=noise,
#         )
#
#     raise ValueError(f"Unknown label kind={kind!r}.")
