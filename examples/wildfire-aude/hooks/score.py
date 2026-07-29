"""Per-run scoring: fit of the simulated scar against the EFFIS truth."""

from __future__ import annotations

import json
import math
import os

# Pass thresholds: the simulated scar must overlap the real one decently
# and land in the right order of magnitude of burnt area.
MIN_IOU = float(os.environ.get("ELODIN_WILDFIRE_MIN_IOU", "0.25"))
MAX_AREA_ERROR = float(os.environ.get("ELODIN_WILDFIRE_MAX_AREA_ERROR", "0.6"))


def _finite(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def post_run(ctx):
    result = {}
    try:
        with open(ctx.run_dir + "/result.json") as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    iou = _finite(result.get("truth_iou"), 0.0)
    area_error = _finite(result.get("area_error_frac"), float("inf"))
    burned_ha = _finite(result.get("burned_ha"), 0.0)
    valid = bool(result)
    return {
        "truth_iou": iou,
        "area_error_frac": area_error,
        "burned_ha": burned_ha,
        "valid": valid,
        "pass": valid and iou >= MIN_IOU and area_error <= MAX_AREA_ERROR,
    }
