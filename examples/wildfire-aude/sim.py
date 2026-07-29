"""Fire-spread simulation of the 2025 Corbieres massif fire (Aude, France).

Model summary (full derivation in README.md):

- The landscape is a (ny, nx) grid of 300 m cells with real elevation
  (Copernicus DEM), real fuel classes (ESA WorldCover) and the real burnt
  scar (EFFIS) as truth reference.
- Fire spreads cell-to-cell along the 8 neighbor directions. The rate of
  spread from a burning neighbor follows a Rothermel-style decomposition
  R = R0 * eta_M * phi_wind * phi_slope, with R0 a per-fuel-class no-wind
  rate, eta_M the Rothermel moisture damping polynomial, and exponential
  wind/slope factors.
- Each cell accumulates "ignition progress" at rate sum_d I_d * R_d / dist_d
  from its burning neighbors; a head fire therefore crosses a cell in
  dist / R seconds, i.e. the front propagates at exactly R.
- Burning cells ramp their intensity up, consume fuel with a per-class
  burnout time, and die when fuel is exhausted. Ember spotting injects
  heat one spotting distance downwind of intense cells.
- Wind is a tramontane profile: documented mean speed and direction with
  a diurnal cycle, slow meander, and gusts synthesized as a deterministic
  sum of sines (seeded per run, so Monte Carlo runs differ).

All per-cell constants are baked into the compiled system as closures; the
only runtime state is fuel / progress / intensity plus scalar metrics.
"""

import os
import typing as ty

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

import reference as ref

# One tick = 20 s of fire time. simulation_rate only sets the editor
# playback clock (60 ticks/s of playback = 1200x real time).
DT_S = 20.0
SIMULATION_RATE_HZ = 60.0
SIM_HOURS = 24.0
DEFAULT_MAX_TICKS = int(SIM_HOURS * 3600.0 / DT_S)
FIRE_START_HOUR = 16.0  # local time of ignition, 2025-08-05

# Fuel classes: 0 none, 1 grass/agriculture, 2 shrub (garrigue), 3 forest.
R0_MS = np.array([0.0, 0.055, 0.050, 0.022])  # no-wind, dry rate of spread
BURNOUT_S = np.array([np.inf, 600.0, 1800.0, 3600.0])  # fuel residence time
MOISTURE_OF_EXTINCTION = 0.25
INTENSITY_RAMP_S = 120.0
SPOT_INTENSITY_MIN = 0.55
SPOT_IGNITION_S = 600.0  # a landed ember ignites its cell in ~10 min

VIZ_FACTOR = int(os.environ.get("WILDFIRE_VIZ_FACTOR", "3"))
TERRAIN_FACTOR = int(os.environ.get("WILDFIRE_TERRAIN_FACTOR", "2"))
WITH_THRUSTERS = os.environ.get("WILDFIRE_THRUSTERS", "1") != "0"
ELEV_EXAGGERATION = 2.0
VIZ_NY = ref.NY // VIZ_FACTOR
VIZ_NX = ref.NX // VIZ_FACTOR
VIZ_N = VIZ_NY * VIZ_NX

# Default ignition: cell center of the documented start area (D212 between
# Ribaute and Lagrasse).
_IGN_X_KM = (ref.IGNITION_IX + 0.5) * ref.CELL_KM
_IGN_Y_KM = (ref.IGNITION_IY + 0.5) * ref.CELL_KM

# (default, min, max) per Monte Carlo parameter. Weather anchors are the
# documented values: tramontane from ~315 deg, gusts ~60 km/h, 35 degC.
PARAM_TABLE = {
    "wind_speed_kmh": (52.0, 35.0, 75.0),
    "wind_from_deg": (297.0, 285.0, 325.0),
    "gust_std": (0.18, 0.0, 0.4),
    "fuel_moisture": (0.08, 0.03, 0.15),
    # Spread-model calibration knobs.
    "r0_scale": (0.85, 0.4, 2.0),
    "k_wind": (0.22, 0.08, 0.35),
    "k_slope": (3.0, 0.5, 6.0),
    "spot_rate": (0.15, 0.0, 0.6),
    "spot_dist_km": (0.9, 0.3, 2.0),
    # Ignition location (km, scene frame).
    "ignition_east_km": (_IGN_X_KM, 0.5, 33.5),
    "ignition_north_km": (_IGN_Y_KM, 0.5, 20.5),
    # Optional firebreak: a cleared strip (fuel removed before the fire).
    "firebreak_on": (0.0, 0.0, 1.0),
    "firebreak_east_km": (13.0, 0.0, 34.0),
    "firebreak_north_km": (12.0, 0.0, 21.0),
    "firebreak_length_km": (5.0, 0.5, 15.0),
    "firebreak_bearing_deg": (45.0, 0.0, 180.0),
    "firebreak_width_m": (250.0, 50.0, 600.0),
    # Gust/spotting noise seed (dispersed by the campaign sampler).
    "noise_seed": (1.0, 0.0, 1e6),
}
PARAMS = el.monte_carlo.params_spec(
    **{
        name: el.monte_carlo.Param(float, default=d, min=lo, max=hi)
        for name, (d, lo, hi) in PARAM_TABLE.items()
    }
)

N = ref.NY * ref.NX


def _flat_component(name: str, size: int):
    return ty.Annotated[
        jax.Array,
        el.Component(name, el.ComponentType(el.PrimitiveType.F64, (size,))),
    ]


FuelFrac = _flat_component("fuel", N)
Progress = _flat_component("progress", N)
Intensity = _flat_component("intensity", N)
VizFlame = _flat_component("viz_flame", VIZ_N)
VizChar = _flat_component("viz_char", VIZ_N)
SimTime = _flat_component("t_s", 1)
BurnedHa = _flat_component("burned_ha", 1)
TruthHa = _flat_component("truth_ha", 1)
TruthIou = _flat_component("truth_iou", 1)
ActiveHa = _flat_component("active_ha", 1)
WindKmh = _flat_component("wind_kmh", 1)
WindMs = ty.Annotated[
    jax.Array,
    el.Component(
        "wind_ms",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "e,n,u"},
    ),
]

# 8 neighbor offsets (dy, dx), row 0 = south, col 0 = west.
NEIGHBORS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _shift(a: jax.Array, dy: int, dx: int) -> jax.Array:
    """out[y, x] = a[y + dy, x + dx] (wrapped edges are masked separately)."""
    return jnp.roll(a, shift=(-dy, -dx), axis=(0, 1))


def _edge_mask(dy: int, dx: int) -> np.ndarray:
    """0 where `_shift(a, dy, dx)` would wrap around the grid, else 1."""
    mask = np.ones((ref.NY, ref.NX))
    if dy > 0:
        mask[ref.NY - dy :, :] = 0.0
    elif dy < 0:
        mask[:-dy, :] = 0.0
    if dx > 0:
        mask[:, ref.NX - dx :] = 0.0
    elif dx < 0:
        mask[:, :-dx] = 0.0
    return mask


def moisture_damping(moisture: float) -> float:
    """Rothermel (1972) moisture damping coefficient eta_M."""
    r = min(max(moisture / MOISTURE_OF_EXTINCTION, 0.0), 1.0)
    return max(1.0 - 2.59 * r + 5.11 * r**2 - 3.52 * r**3, 0.0)


def firebreak_mask(p: dict) -> np.ndarray:
    """1 where fuel is untouched, 0 inside the cleared strip."""
    keep = np.ones((ref.NY, ref.NX))
    if p["firebreak_on"] < 0.5:
        return keep
    x, y = ref.cell_centers_km()
    bearing = np.radians(p["firebreak_bearing_deg"])
    ux, uy = np.sin(bearing), np.cos(bearing)  # along-strip unit vector
    dx, dy = x - p["firebreak_east_km"], y - p["firebreak_north_km"]
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux
    inside = (np.abs(along) <= p["firebreak_length_km"] / 2.0) & (
        np.abs(across) <= p["firebreak_width_m"] / 2000.0
    )
    keep[inside] = 0.0
    return keep


def _gust_series(rng: np.random.Generator, n_modes: int = 6):
    """Coefficients of a deterministic sum-of-sines gust signal g(t)."""
    periods = rng.uniform(300.0, 3600.0, n_modes)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    amps = rng.uniform(0.5, 1.0, n_modes)
    amps /= np.sqrt((amps**2).sum() / 2.0)  # roughly unit-variance signal
    return jnp.asarray(2.0 * np.pi / periods), jnp.asarray(phases), jnp.asarray(amps)


def build(params) -> tuple[el.World, el.System]:
    p = {name: float(params.get(name, spec[0])) for name, spec in PARAM_TABLE.items()}
    rng = np.random.default_rng(int(p["noise_seed"]) % 2**32)

    # --- Baked landscape constants ----------------------------------------
    keep = firebreak_mask(p)
    fuel0_np = ref.BURNABLE_FRAC * (ref.FUEL_CLASS > 0) * keep
    r0_np = R0_MS[ref.FUEL_CLASS] * ref.BURNABLE_FRAC * keep * p["r0_scale"]
    burn_rate_np = np.where(ref.FUEL_CLASS > 0, 1.0 / BURNOUT_S[ref.FUEL_CLASS], 0.0)
    eta_m = moisture_damping(p["fuel_moisture"])

    # Per-direction static kernel K_d = mask * phi_slope * R0(source) / dist:
    # everything in the neighbor heat sum that does not depend on time.
    kernel_list, unit_vecs = [], []
    for dy, dx in NEIGHBORS:
        dist_m = ref.CELL_M * float(np.hypot(dy, dx))
        z_src = np.roll(ref.ELEVATION_M, shift=(-dy, -dx), axis=(0, 1))
        slope = np.clip((ref.ELEVATION_M - z_src) / dist_m, -0.6, 0.6)
        phi_slope = np.exp(p["k_slope"] * slope)
        r0_src = np.roll(r0_np, shift=(-dy, -dx), axis=(0, 1))
        kernel_list.append(_edge_mask(dy, dx) * phi_slope * r0_src * eta_m / dist_m)
        norm = float(np.hypot(dy, dx))
        unit_vecs.append((-dx / norm, -dy / norm))  # from source cell to us
    kernels = jnp.asarray(np.stack(kernel_list))
    unit_east = jnp.asarray([u[0] for u in unit_vecs])
    unit_north = jnp.asarray([u[1] for u in unit_vecs])

    fuel0 = jnp.asarray(fuel0_np)
    fuel0_safe = jnp.asarray(np.maximum(fuel0_np, 1e-6))
    burn_rate = jnp.asarray(burn_rate_np)
    truth = jnp.asarray(ref.TRUTH_MASK.astype(np.float64))

    # --- Ember spotting ----------------------------------------------------
    # Cells drawn as ember sources leak heat one spotting distance downwind
    # (static shift along the mean wind) while they burn intensely.
    ember = jnp.asarray((rng.random((ref.NY, ref.NX)) < p["spot_rate"]).astype(np.float64))
    to_dir = np.radians(p["wind_from_deg"] + 180.0)
    spot_dx = int(round(p["spot_dist_km"] * np.sin(to_dir) / ref.CELL_KM))
    spot_dy = int(round(p["spot_dist_km"] * np.cos(to_dir) / ref.CELL_KM))
    spot_edge = jnp.asarray(_edge_mask(-spot_dy, -spot_dx))

    # --- Wind profile --------------------------------------------------------
    w_omega, w_phase, w_amp = _gust_series(rng)
    d_omega, d_phase, d_amp = _gust_series(rng)
    base_speed_ms = p["wind_speed_kmh"] / 3.6

    def wind_at(t_s: jax.Array) -> tuple[jax.Array, jax.Array]:
        hour_angle = 2.0 * jnp.pi * (t_s / 3600.0) / 24.0
        diurnal = 0.75 + 0.25 * jnp.cos(hour_angle)
        gust = jnp.sum(w_amp * jnp.sin(w_omega * t_s + w_phase))
        speed = base_speed_ms * diurnal * jnp.maximum(1.0 + p["gust_std"] * gust, 0.15)
        meander = 12.0 * jnp.sin(2.0 * jnp.pi * t_s / (6.0 * 3600.0))
        wobble = 6.0 * jnp.sum(d_amp * jnp.sin(d_omega * t_s + d_phase))
        to_rad = jnp.radians(p["wind_from_deg"] + 180.0 + meander + wobble)
        return speed * jnp.sin(to_rad), speed * jnp.cos(to_rad)

    # --- Initial state --------------------------------------------------------
    progress0 = np.zeros((ref.NY, ref.NX))
    ix = int(np.clip(p["ignition_east_km"] / ref.CELL_KM, 1, ref.NX - 2))
    iy = int(np.clip(p["ignition_north_km"] / ref.CELL_KM, 1, ref.NY - 2))
    progress0[iy : iy + 2, ix : ix + 2] = 1.05

    # --- The tick ---------------------------------------------------------------
    @el.map
    def step(
        fuel: FuelFrac, progress: Progress, intensity: Intensity, t: SimTime
    ) -> tuple[
        FuelFrac,
        Progress,
        Intensity,
        SimTime,
        WindMs,
        BurnedHa,
        TruthIou,
        ActiveHa,
        WindKmh,
        VizFlame,
        VizChar,
    ]:
        f = fuel.reshape(ref.NY, ref.NX)
        prog = progress.reshape(ref.NY, ref.NX)
        inten = intensity.reshape(ref.NY, ref.NX)
        t_s = t[0]

        wind_e, wind_n = wind_at(t_s)

        # Heat received from the 8 neighbors: I_src * R(src->cell) / dist.
        dot = unit_east * wind_e + unit_north * wind_n
        phi_wind = jnp.exp(p["k_wind"] * jnp.maximum(dot, 0.0) - 0.06 * jnp.maximum(-dot, 0.0))
        heat = sum(
            phi_wind[d] * kernels[d] * _shift(inten, dy, dx) for d, (dy, dx) in enumerate(NEIGHBORS)
        )
        # Ember spotting: heat arrives one spot distance downwind of sources.
        embers = _shift(inten * ember * (inten > SPOT_INTENSITY_MIN), -spot_dy, -spot_dx)
        heat = heat + spot_edge * embers / SPOT_IGNITION_S

        prog = jnp.minimum(prog + DT_S * heat * (fuel0 > 0.0), 1.5)
        ignited = prog >= 1.0

        # Intensity ramps toward a fuel-limited target; fuel burns out.
        target = ignited * jnp.sqrt(jnp.clip(f / fuel0_safe, 0.0, 1.0))
        inten = inten + (DT_S / INTENSITY_RAMP_S) * (target - inten)
        f = jnp.clip(f - DT_S * inten * burn_rate, 0.0, 1.0)

        burned = ignited.astype(jnp.float64)
        burned_ha = jnp.sum(burned) * ref.CELL_HA
        inter = jnp.sum(burned * truth)
        union = jnp.sum(jnp.maximum(burned, truth))
        iou = inter / jnp.maximum(union, 1.0)
        active_ha = jnp.sum(inten > 0.2) * ref.CELL_HA
        speed_kmh = jnp.sqrt(wind_e**2 + wind_n**2) * 3.6

        # Down-sampled grids driving the 3D scene (per-block pooling).
        def blocks(a: jax.Array) -> jax.Array:
            crop = a[: VIZ_NY * VIZ_FACTOR, : VIZ_NX * VIZ_FACTOR]
            return crop.reshape(VIZ_NY, VIZ_FACTOR, VIZ_NX, VIZ_FACTOR)

        viz_flame = blocks(inten).max(axis=(1, 3)).reshape(-1)
        char = (1.0 - jnp.clip(f / fuel0_safe, 0.0, 1.0)) * burned * (fuel0 > 0.0)
        viz_char = blocks(char).mean(axis=(1, 3)).reshape(-1)

        return (
            f.reshape(-1),
            prog.reshape(-1),
            inten.reshape(-1),
            jnp.array([t_s + DT_S]),
            jnp.array([wind_e, wind_n, 0.0]),
            jnp.array([burned_ha]),
            jnp.array([iou]),
            jnp.array([active_ha]),
            jnp.array([speed_kmh]),
            viz_flame,
            viz_char,
        )

    # --- World -------------------------------------------------------------------
    world = el.World()
    world.spawn(
        [
            el.C(FuelFrac, jnp.asarray(fuel0_np.reshape(-1))),
            el.C(Progress, jnp.asarray(progress0.reshape(-1))),
            el.C(Intensity, jnp.zeros(N)),
            el.C(SimTime, jnp.array([0.0])),
            el.C(WindMs, jnp.zeros(3)),
            el.C(BurnedHa, jnp.array([0.0])),
            el.C(TruthHa, jnp.array([float(ref.TRUTH_MASK.sum() * ref.CELL_HA)])),
            el.C(TruthIou, jnp.array([0.0])),
            el.C(ActiveHa, jnp.array([0.0])),
            el.C(WindKmh, jnp.array([0.0])),
            el.C(VizFlame, jnp.zeros(VIZ_N)),
            el.C(VizChar, jnp.zeros(VIZ_N)),
        ],
        name="fire",
    )
    world.schematic(build_schematic(p), "wildfire-aude.kdl")
    return world, step


# --- Visualization -----------------------------------------------------------

FUEL_COLORS = {
    0: (126, 122, 128),  # bare / built-up
    1: (196, 186, 108),  # grass and crops
    2: (128, 138, 66),  # garrigue
    3: (44, 84, 42),  # forest
    4: (52, 96, 148),  # water
}


def _hillshade() -> np.ndarray:
    """Lambertian hillshade of the DEM, sun from the northwest, 45 deg up."""
    dz_dy, dz_dx = np.gradient(ref.ELEVATION_M * ELEV_EXAGGERATION, ref.CELL_M)
    normal = np.dstack([-dz_dx, -dz_dy, np.ones_like(dz_dx)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    azimuth, altitude = np.radians(315.0), np.radians(45.0)
    sun = np.array(
        [
            np.sin(azimuth) * np.cos(altitude),
            np.cos(azimuth) * np.cos(altitude),
            np.sin(altitude),
        ]
    )
    return np.clip(normal @ sun, 0.0, 1.0)


def _viz_blocks(factor: int) -> list[dict]:
    """Static per-block description of the terrain (position, size, color)."""
    cell_km = ref.CELL_KM * factor
    shade = _hillshade()
    out = []
    for by in range(ref.NY // factor):
        for bx in range(ref.NX // factor):
            ys = slice(by * factor, (by + 1) * factor)
            xs = slice(bx * factor, (bx + 1) * factor)
            elev = float(ref.ELEVATION_M[ys, xs].mean())
            fuel = int(np.bincount(ref.FUEL_CLASS[ys, xs].ravel(), minlength=4).argmax())
            base = FUEL_COLORS[4 if elev <= 0.0 else fuel]
            light = 0.45 + 0.65 * float(shade[ys, xs].mean())
            out.append(
                dict(
                    x=(bx + 0.5) * cell_km,
                    y=(by + 0.5) * cell_km,
                    top=max(0.03, elev * ELEV_EXAGGERATION / 1000.0),
                    color=tuple(min(255, int(round(c * light))) for c in base),
                    burnable=fuel > 0 and elev > 0.0,
                )
            )
    return out


def build_schematic(p: dict) -> str:
    cell_km = ref.CELL_KM * VIZ_FACTOR
    side = cell_km * 0.96
    cx, cy = ref.NX * ref.CELL_KM / 2.0, ref.NY * ref.CELL_KM / 2.0
    # Plume tilt from the wind/updraft ratio: a ~50 km/h tramontane against a
    # ~10 m/s convective rise gives tan(tilt) ~ 1.2, i.e. ~50 deg off vertical.
    # (A straight-up direction would also be a degenerate rotation arc.)
    wind_to = np.radians(p["wind_from_deg"] + 180.0)
    tilt = np.clip(p["wind_speed_kmh"] / 3.6 / 12.0, 0.35, 1.8)
    lean_e, lean_n = tilt * np.sin(wind_to), tilt * np.cos(wind_to)
    plume_dir = f"({lean_e:.3f}, {lean_n:.3f}, 1.0)"
    lines = [
        "coordinate frame=ENU",
        # Physical sun + exposure, transcribed from the pyrotechnique authoring
        # scene (assets/scenes/wildfire.scene.ron there) so the .effect files
        # look here exactly as they were tuned. Azimuth/elevation match the
        # hillshade baked into the terrain colours. No `atmosphere` node: Bevy's
        # procedural sky is parameterised in metres and this scene is in km.
        # Shadows off — several thousand terrain boxes would cost more than the
        # relief cue is worth (the hillshade already carries it).
        "environment {",
        "    sun azimuth=315.0 elevation=45.0 illuminance=110000.0 shadows=#false",
        "    ambient scale=0.35",
        "}",
        "hsplit {",
        "    tabs share=0.72 {",
        f'        viewport name="Corbières" pos="(0,0,0,1, {cx - 6.0:.2f}, {cy - 12.0:.2f}, 7.0)"'
        f' look_at="(0,0,0,1, {cx:.2f}, {cy:.2f}, 0.4)" fov=42.0 hdr=#true ev100=14.0'
        " active=#true {",
        '            bloom preset="old_school" intensity=0.5 threshold=0.6 threshold_softness=0.25',
        "        }",
        f'        viewport name="Carte" pos="(0,0,0,1, {cx - 0.4:.2f}, {cy - 2.0:.2f}, 30.0)"'
        f' look_at="(0,0,0,1, {cx:.2f}, {cy:.2f}, 0.0)" fov=45.0 hdr=#true ev100=14.0 {{',
        '            bloom preset="old_school" intensity=0.4 threshold=0.65 threshold_softness=0.2',
        "        }",
        "    }",
        "    vsplit share=0.28 {",
        '        graph "fire.burned_ha, fire.truth_ha" name="Surface brûlée vs EFFIS (ha)"',
        '        graph "fire.truth_iou" name="IoU vs périmètre EFFIS"',
        '        graph "fire.wind_kmh" name="Vent (km/h)"',
        '        graph "fire.active_ha" name="Front actif (ha)"',
        "    }",
        "}",
        # Surrounding countryside stretching to the horizon.
        f'object_3d "(0,0,0,1, {cx:.2f}, {cy:.2f}, -0.012)" {{',
        "    plane width=1200 depth=1200 { color 96 96 62 }",
        "}",
    ]

    # Terrain at a finer grain than the fire overlays: static boxes are cheap.
    terrain_cell = ref.CELL_KM * TERRAIN_FACTOR
    terrain_side = terrain_cell * 0.98
    for blk in _viz_blocks(TERRAIN_FACTOR):
        x, y, top = blk["x"], blk["y"], blk["top"]
        r, g, b = blk["color"]
        lines += [
            f'object_3d "(0,0,0,1, {x:.3f}, {y:.3f}, {top / 2.0:.4f})" {{',
            f"    box x={terrain_side:.3f} y={terrain_side:.3f} z={top:.4f}"
            f" {{ color {r} {g} {b} }}",
            "}",
        ]

    for i, blk in enumerate(_viz_blocks(VIZ_FACTOR)):
        if not blk["burnable"]:
            continue
        x, y, top = blk["x"], blk["y"], blk["top"]
        flame = f"fire.viz_flame[{i}]"
        char = f"fire.viz_char[{i}]"
        lines += [
            # Charred overlay: a flat dark disc that grows as fuel burns.
            f'object_3d "(0,0,0,1, {x:.3f}, {y:.3f}, {top:.4f})" {{',
            f'    ellipsoid scale="(0.001 + {0.52 * side:.3f} * {char},'
            f' 0.001 + {0.52 * side:.3f} * {char}, 0.001 + 0.02 * {char})" {{ color 26 23 20 225 }}',
            "}",
        ]
        flame_obj = [
            # Ember bed: a flat incandescent patch on the ground. Deliberately
            # squashed (~40 m tall) — the flame particles are authored at real
            # flame length, so a tall glowing balloon here would fight them.
            # This is what carries the fire line at 15 km, lifted by bloom.
            f'object_3d "(0,0,0,1, {x:.3f}, {y:.3f}, {top:.4f})" {{',
            f'    ellipsoid scale="(0.001 + {0.40 * side:.3f} * {flame},'
            f' 0.001 + {0.40 * side:.3f} * {flame}, 0.001 + 0.04 * {flame})"'
            " { color 255 96 12 235 }",
        ]
        if WITH_THRUSTERS:
            # Hanabi particle effects (flames + smoke + sparks), throttled by
            # the cell's live intensity. Authored in assets/effects/wildfire/.
            flame_obj += [
                f'    thruster name="fl{i}" effect="effects/wildfire/flame.effect"'
                f' position="(0, 0, 0)" direction="{plume_dir}" intensity="{flame}" cutoff=0.08 {{',
                '        effect "effects/wildfire/smoke.effect"',
                '        effect "effects/wildfire/sparks.effect"',
                "    }",
            ]
        flame_obj.append("}")
        lines += flame_obj

    # Real EFFIS perimeter as a fence of dark-red posts (the target result).
    # Real EFFIS perimeter as a dotted red line: 50 m beads every ~150 m along
    # the ring. Dots read as an outline from any viewing angle, where vertical
    # posts read as buildings. Each bead sits on the mean elevation of the
    # terrain block it falls in — the same value the boxes are drawn at — so
    # the line hugs the visible surface instead of floating or sinking into it.
    for px, py in ref.truth_perimeter_km(step=6):
        gx = min(max(int(px / ref.CELL_KM), 0), ref.NX - 1)
        gy = min(max(int(py / ref.CELL_KM), 0), ref.NY - 1)
        bx, by = gx // TERRAIN_FACTOR, gy // TERRAIN_FACTOR
        block = ref.ELEVATION_M[
            by * TERRAIN_FACTOR : (by + 1) * TERRAIN_FACTOR,
            bx * TERRAIN_FACTOR : (bx + 1) * TERRAIN_FACTOR,
        ]
        z = max(0.03, float(block.mean()) * ELEV_EXAGGERATION / 1000.0)
        lines += [
            f'object_3d "(0,0,0,1, {px:.3f}, {py:.3f}, {z + 0.03:.4f})" {{',
            "    sphere radius=0.05 { color 235 45 45 }",
            "}",
        ]

    # Ignition marker + wind arrow above the scene.
    lines += [
        f'object_3d "(0,0,0,1, {p["ignition_east_km"]:.3f}, {p["ignition_north_km"]:.3f}, 1.35)" {{',
        "    sphere radius=0.16 { color 255 220 40 }",
        "}",
        f'vector_arrow "fire.wind_ms" origin="(0,0,0,1, {cx:.2f}, {cy + 7.0:.2f}, 4.2)"'
        ' scale=0.16 name="Tramontane" show_name=#true {',
        "    color 120 190 255",
        "}",
    ]
    if p["firebreak_on"] >= 0.5:
        bearing = np.radians(p["firebreak_bearing_deg"])
        ux, uy = np.sin(bearing), np.cos(bearing)
        fx, fy = p["firebreak_east_km"], p["firebreak_north_km"]
        length = p["firebreak_length_km"]
        for s in np.linspace(-length / 2.0, length / 2.0, max(int(length / 0.45), 2)):
            lines += [
                f'object_3d "(0,0,0,1, {fx + s * ux:.3f}, {fy + s * uy:.3f}, 0.55)" {{',
                "    box x=0.1 y=0.1 z=1.1 { color 90 200 255 }",
                "}",
            ]
    return "\n".join(lines)
