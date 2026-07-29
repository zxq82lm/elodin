# Wildfire Aude 2025 — Corbières massif fire

A fire-spread simulation of the largest French wildfire in decades, grounded
entirely in open data, visualized in the Elodin editor, and calibrated
against the real burnt-area perimeter with `elodin monte-carlo`.

**The event.** On 2025-08-05 around 16:00 local, a fire started on the
roadside of the D212 between Ribaute and Lagrasse (Aude). Pushed by a
tramontane blowing from the northwest with gusts near 60 km/h in a 35 °C
airmass, it ran ~20 km southeast through the Corbières massif in one night,
ultimately affecting ~16 000 ha across 16 communes (11 391 ha of burnt
vegetation mapped by satellite). It was declared extinguished on 2025-08-28.

## Run it

```bash
nix develop
just install

# 3D scene + graphs (playback is ~1200x real time: 24 h of fire in ~72 s)
elodin editor examples/wildfire-aude/main.py

# Headless single run
elodin run examples/wildfire-aude/main.py

# Calibration campaign: 48 dispersed runs scored against the real perimeter
elodin monte-carlo run examples/wildfire-aude/main.py \
  --campaign examples/wildfire-aude/campaign.toml \
  --spec examples/wildfire-aude/spec.toml \
  --out dbs/wildfire-campaign
```

In the scene: terrain colored by real land cover and hillshaded from the
real DEM, the yellow sphere marks the documented ignition point, dark-red
posts fence the real EFFIS burnt perimeter (the target), the active front
glows with flame/smoke/spark particle plumes leaning downwind, dark discs
are the charred area, and the blue arrow tracks the simulated tramontane.

The particle effects are bevy_hanabi `.effect` files in
`assets/effects/wildfire/` (flame, smoke, sparks), the same format authored
by [pyrotechnique](https://github.com/elodin-sys/pyrotechnique); edit them
there and drop the exports back in.

## Open data (all vendored, with provenance)

| Layer | Source | File |
|---|---|---|
| Burnt-area perimeter (truth) | [EFFIS](https://forest-fire.emergency.copernicus.eu/) burnt area id 276934, CC-BY | `data/effis_ba_276934.json` (verbatim API response) |
| Elevation | [Copernicus DEM GLO-30](https://registry.opendata.aws/copernicus-dem/) (ESA), 30 m | resampled into `data/grids.npz` |
| Land cover / fuels | [ESA WorldCover 10 m v200](https://esa-worldcover.org/) (2021) | aggregated into `data/grids.npz` |
| Weather anchors | Météo-France red-alert forecast for 2025-08-05, press/ECHO reports | constants in `reference.py` |

`data/prep_data.py` regenerates the derived grids from the primary sources
(one-time, network required). `reference.py` loads them, converts to the
scene frame, and `python reference.py` runs the sanity checks: rasterized
scar area within 3 % of EFFIS, perimeter inside the domain, ignition cell
burnable, woody-fuel fraction inside the scar consistent with the EFFIS
per-class breakdown.

The domain is a 114 x 70 grid of 300 m cells (34.2 x 21.0 km), ENU, one
scene unit = 1 km, origin at the domain's southwest corner
(lon 2.55°E, lat 42.95°N).

## The model

This is a quasi-physical cellular spread model in the lineage used by
operational tools (FARSITE, ForeFire, FSim all build on the same
ingredients: Rothermel-style rate of spread + front propagation). Everything
is JAX on one `(70, 114)` grid; a 24 h fire (4 320 ticks of 20 s) runs in
~10 s on a laptop CPU.

### Rate of spread

The spread rate from a burning cell toward its neighbor in direction $d$:

$$R_d = R_0 \cdot \eta_M \cdot \phi_w(d) \cdot \phi_s(d)$$

- $R_0$: no-wind, dry rate of spread per fuel class (grass 0.055 m/s, shrub
  0.050, forest 0.022, scaled by the cell's burnable fraction and the
  `r0_scale` calibration knob).
- $\eta_M$: Rothermel (1972) moisture damping,
  $\eta_M = 1 - 2.59 r + 5.11 r^2 - 3.52 r^3$ with
  $r = M / M_x$ and extinction moisture $M_x = 0.25$.
- $\phi_w = \exp(k_w \, \max(0, \vec{W} \cdot \hat{u}_d))$: exponential wind
  alignment factor ($\hat u_d$ = spread direction, $\vec W$ = wind vector);
  a small $\exp(-0.06 \max(0, -\vec W \cdot \hat u_d))$ term damps backing
  fire. At 52 km/h this gives a head/flank ratio of ~25, typical of
  wind-driven shrub fires.
- $\phi_s = \exp(k_s \, \tan\theta_d)$: slope factor from the real DEM
  (upslope spread is faster; van Wagner's rule of thumb, doubling every
  ~10°, corresponds to $k_s \approx 4$).

### Front propagation

Each cell accumulates *ignition progress* from its 8 neighbors:

$$\frac{dp}{dt} = \sum_d I_d \, \frac{R_d}{\Delta_d}, \qquad p \ge 1
\Rightarrow \text{ignition}$$

where $I_d$ is the neighbor's burning intensity and $\Delta_d$ the
center-to-center distance. A head fire therefore crosses a cell in exactly
$\Delta / R$ seconds — the front propagates at the Rothermel rate by
construction. Everything static (slope factors, source $R_0$, edge masks,
$1/\Delta$) is prebaked into 8 per-direction kernels, so a tick is just 8
`jnp.roll` + multiply-accumulate.

Ignited cells ramp intensity with a 2-minute time constant toward a
fuel-limited target, and consume fuel with per-class residence times
(grass 10 min, shrub 30 min, forest 60 min); intensity dies as fuel runs
out. Ember spotting draws a random subset of cells as ember sources which
inject heat one spotting distance downwind while burning intensely.

### Wind

The tramontane is modeled as: documented mean direction and speed, a
diurnal cycle (peak at ignition time, minimum before dawn), a slow 6-hour
directional meander, and gusts synthesized as a deterministic sum of six
sines (unit variance, periods 5–60 min). All noise is seeded from the
`noise_seed` Monte Carlo parameter — runs are perfectly reproducible, and
the campaign disperses the seed.

## Monte Carlo calibration

`spec.toml` disperses what is genuinely uncertain: wind speed/direction/
gustiness, dead-fuel moisture, the model knobs ($R_0$ scale, $k_w$,
spotting), the exact ignition point, and the gust realization. Each run
scores itself against the EFFIS scar in `main.py` and emits:

- `truth_iou` — intersection-over-union of simulated vs real burnt area,
- `burned_ha` and `area_error_frac`,
- `active_ha_end`.

`hooks/score.py` turns these into pass/fail; `hooks/report.py` prints fit
statistics and the **best-fit parameter sets** — the calibration signal.
A representative 48-run campaign finishes in ~60 s and produced:

```text
IoU vs EFFIS scar   mean 0.216   p50 0.271   best 0.512
burned area (ha)    p50 18950    p95 35453   truth 11391
best fit: wind_from=301° wind=45 km/h r0_scale=1.04 k_wind=0.218 moisture=0.075
```

The defaults in `sim.py` are a hand-calibrated robust point (IoU 0.38,
15 400 ha vs 16 000 ha officially affected). Two honest lessons from the
campaign, worth knowing before trusting any single run:

- **Percolation sensitivity.** Total burnt area is extremely steep in
  $R_0 \cdot \eta_M$: 25 % less base ROS collapses the fire from 15 000 to
  3 000 ha. Real fire models share this threshold behavior; it is why
  distributions, not single runs, are the deliverable.
- **Seed sensitivity.** The campaign's best run (IoU 0.51) owes part of its
  fit to its particular gust realization: replayed with a different
  `noise_seed`, the same parameters give IoU 0.24. Calibrate on the
  distribution across seeds, not on one lucky draw.

Known model limitation: the real scar has a northern lobe (toward Lagrasse)
that the simulation never reproduces — it likely burned before the
tramontane fully established, and this model has no time-varying synoptic
wind shift. Adding one (direction ramp over the first hours) is a good
first contribution.

## Acting optimally: the firebreak

The original motivation: *if you can simulate it, you can Monte Carlo where
to act.* The `firebreak_*` parameters carve a fuel-free strip (a
tactical firebreak) into the landscape before ignition:

```bash
# One run with a 5 km firebreak across the corridor
# (edit spec.toml to disperse its position and compare expected burnt area)
python - <<'EOF'
import json
ctx = dict(run_id="fb", seed=1, db_path=None, db_addr=None, cache_dir=None,
           run_dir="/tmp/fb", params={"firebreak_on": 1.0}, meta={}, slots={})
open("/tmp/fb.json", "w").write(json.dumps(ctx))
EOF
mkdir -p /tmp/fb && ELODIN_MONTE_CARLO_CONTEXT=/tmp/fb.json \
  elodin run examples/wildfire-aude/main.py && cat /tmp/fb/result.json
```

To find the *optimal* placement, sweep `firebreak_east_km` /
`firebreak_north_km` in `spec.toml` while dispersing the weather, and pick
the placement minimizing expected `burned_ha` (and its p95 — in crisis
management the tail matters more than the mean).

## Files

| File | Role |
|---|---|
| `main.py` | Entry point: params, run, Monte Carlo result emission |
| `sim.py` | Model + world + generated KDL scene |
| `reference.py` | Truth data loading, scene frame, sanity checks |
| `data/prep_data.py` | One-time derivation of `grids.npz` from open sources |
| `spec.toml` / `campaign.toml` | Campaign sampling and orchestration |
| `hooks/score.py` / `hooks/report.py` | Per-run scoring and campaign report |

Environment knobs: `ELODIN_WILDFIRE_MAX_TICKS` (default 4320),
`WILDFIRE_VIZ_FACTOR` (fire-overlay block size, default 3 cells),
`WILDFIRE_TERRAIN_FACTOR` (terrain block size, default 2 cells),
`WILDFIRE_THRUSTERS=0` (disable particle plumes), `ELODIN_BACKEND`
(defaults to `jax-cpu`; the grid workload currently trips a Cranelift
codegen limit on arm64).

## Honest scope

This is a demonstrator of the method — real terrain, real fuels, a
defensible spread model, and a truth-scored campaign — not an operational
tool. ForeFire (CNRS, Université de Corse) and FARSITE embody twenty years
of fuel-model calibration; use them for anything real. What this example
shows is the full loop: open data → physics on a grid → editor
visualization → Monte Carlo calibration → intervention optimization.
