#!/usr/bin/env uv run
"""Entry point: single runs, `elodin editor`, and Monte Carlo campaigns."""

from __future__ import annotations

import os

import elodin as el

import reference as ref
from sim import DEFAULT_MAX_TICKS, DT_S, PARAMS, SIMULATION_RATE_HZ, build

MAX_TICKS_ENV = "ELODIN_WILDFIRE_MAX_TICKS"
TELEMETRY_RATE_HZ = 6.0  # record 1 tick in 10 (one DB frame per 200 s of fire)

params = el.monte_carlo.params(PARAMS)
world, system = build(params)
max_ticks = int(os.environ.get(MAX_TICKS_ENV, str(DEFAULT_MAX_TICKS)))
# Campaign runs must exit at max_ticks; otherwise stay alive so the editor
# keeps its connection and the full timeline can be replayed.
in_campaign = bool(os.environ.get("ELODIN_MONTE_CARLO_CONTEXT"))
result_emitted = False


def post_step(tick: int, ctx: el.StepContext) -> None:
    global result_emitted
    if result_emitted or tick < max_ticks - 1:
        return
    reads = ctx.component_batch_operation(
        reads=["fire.burned_ha", "fire.truth_iou", "fire.active_ha"]
    )
    burned_ha = float(reads["fire.burned_ha"][0])
    truth_iou = float(reads["fire.truth_iou"][0])
    active_ha = float(reads["fire.active_ha"][0])
    if os.environ.get("ELODIN_MONTE_CARLO_CONTEXT"):
        el.monte_carlo.result(
            burned_ha=burned_ha,
            truth_iou=truth_iou,
            area_error_frac=abs(burned_ha - ref.EFFIS_BURNED_HA) / ref.EFFIS_BURNED_HA,
            active_ha_end=active_ha,
            sim_hours=tick * DT_S / 3600.0,
        )
    result_emitted = True


world.run(
    system,
    simulation_rate=SIMULATION_RATE_HZ,
    telemetry_rate=TELEMETRY_RATE_HZ,
    default_playback_speed=1.0,
    max_ticks=max_ticks,
    db_path=params.db_path or os.environ.get("ELODIN_DB_PATH"),
    post_step=post_step,
    interactive=not in_campaign,
    log_level="warn",
    # The grid workload trips a Cranelift codegen limit on arm64 (jump range
    # assertion in cranelift-codegen); XLA handles it fine.
    backend=os.environ.get("ELODIN_BACKEND", "jax-cpu"),
)
