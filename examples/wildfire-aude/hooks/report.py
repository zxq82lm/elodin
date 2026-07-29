"""Campaign report: fit statistics and the best-fit parameter sets.

The best-fit block is the calibration signal: narrow spec.toml around the
reported parameters and re-run to converge on the EFFIS scar.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

EFFIS_BURNED_HA = 11_391.0


def _to_float(value, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def post_campaign(ctx):
    out_dir = Path(ctx.out_dir)
    rows = []
    results = out_dir / "results.csv"
    if results.exists():
        with results.open(newline="") as f:
            rows = list(csv.DictReader(f))
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed") == "true")
    ious = [v for row in rows if (v := _to_float(row.get("truth_iou"))) is not None]
    burned = [v for row in rows if (v := _to_float(row.get("burned_ha"))) is not None]

    scored = []
    for row in rows:
        iou = _to_float(row.get("truth_iou"))
        if iou is None:
            continue
        context_path = out_dir / "runs" / row.get("run_id", "") / "post_run_context.json"
        try:
            params = json.loads(context_path.read_text()).get("params", {})
        except (OSError, json.JSONDecodeError):
            params = {}
        scored.append((iou, _to_float(row.get("burned_ha"), 0.0), row.get("run_id", ""), params))
    scored.sort(reverse=True, key=lambda item: item[0])

    summary = json.loads(Path(ctx.summary).read_text())
    lines = [
        "Wildfire Aude 2025 — campaign fit vs EFFIS burnt area",
        "=====================================================",
        "",
        f"runs completed: {total}   passed: {passed}/{total}",
        f"failed: {summary.get('failed', 0)}   invalid: {summary.get('invalid', 0)}",
        "",
        f"IoU vs EFFIS scar   mean {_fmt(sum(ious) / len(ious) if ious else None)}"
        f"   p50 {_fmt(_percentile(ious, 0.5))}   best {_fmt(max(ious) if ious else None)}",
        f"burned area (ha)    p50 {_fmt(_percentile(burned, 0.5))}"
        f"   p95 {_fmt(_percentile(burned, 0.95))}   truth {EFFIS_BURNED_HA:.0f}",
        "",
        "Best fits (calibration candidates)",
    ]
    keys = ("wind_from_deg", "wind_speed_kmh", "r0_scale", "k_wind", "fuel_moisture", "spot_rate")
    for iou, burned_ha, run_id, params in scored[:5]:
        text = ", ".join(f"{k}={params[k]:.3g}" for k in keys if k in params)
        lines.append(f"  {run_id}: iou={iou:.3f} burned={burned_ha:.0f} ha ({text})")

    report = out_dir / "post_campaign" / "report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    report.write_text(text)
    print(text)
    return {
        "completed": total,
        "passed": passed,
        "best_iou": max(ious) if ious else None,
        "median_burned_ha": _percentile(burned, 0.5),
    }
