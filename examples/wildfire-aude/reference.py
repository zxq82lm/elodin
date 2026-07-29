"""Truth data for the 2025 Corbieres massif fire (Aude, France).

Loads the vendored grids derived by `data/prep_data.py` from open sources
(EFFIS burnt area, Copernicus DEM GLO-30, ESA WorldCover 10 m) and exposes
them in the local scene frame used by the simulation:

- ENU, kilometers, origin at the domain south-west corner;
- grids are (ny, nx) arrays with row 0 = south, column 0 = west.

Only numpy + stdlib so external tools (hooks, plots) can import it too.
Run `python reference.py` to print the profile and the sanity checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# Documented event anchors (Prefecture of Aude / ECHO daily flash / press).
FIRE_START_LOCAL = "2025-08-05T16:00+02:00"
OFFICIAL_BURNED_HA = 16_000  # total affected area over 15+ communes
EFFIS_BURNED_HA = 11_391  # satellite-mapped burnt vegetation
# Meteo-France forecast for 2025-08-05: 35 degC, tramontane (NW) with gusts
# around 60 km/h.
DOC_WIND_FROM_DEG = 315.0
DOC_WIND_KMH = 60.0
DOC_AIR_TEMP_C = 35.0


def _load():
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    grids = np.load(DATA_DIR / "grids.npz")
    return meta, grids


META, _GRIDS = _load()
NX: int = META["nx"]
NY: int = META["ny"]
CELL_M: float = META["cell_m"]
CELL_KM: float = CELL_M / 1000.0
CELL_HA: float = (CELL_M / 100.0) ** 2
ELEVATION_M = _GRIDS["elevation"].astype(np.float64)
FUEL_CLASS = _GRIDS["fuel_class"].astype(np.int64)  # 0 none, 1 grass, 2 shrub, 3 forest
BURNABLE_FRAC = _GRIDS["burnable_frac"].astype(np.float64)
TRUTH_MASK = _GRIDS["truth"].astype(bool)
IGNITION_IX: int = META["ignition_ix"]
IGNITION_IY: int = META["ignition_iy"]


def cell_centers_km() -> tuple[np.ndarray, np.ndarray]:
    """East/north cell-center coordinates in km, shape (ny, nx)."""
    x = (np.arange(NX) + 0.5) * CELL_KM
    y = (np.arange(NY) + 0.5) * CELL_KM
    return np.meshgrid(x, y)


def lonlat_to_km(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (np.asarray(lon) - META["lon_min"]) * META["m_per_deg_lon"] / 1000.0
    y = (np.asarray(lat) - META["lat_min"]) * META["m_per_deg_lat"] / 1000.0
    return x, y


def truth_perimeter_km(step: int = 20) -> np.ndarray:
    """Sub-sampled EFFIS perimeter (outer rings) as (n, 2) east/north km."""
    effis = json.loads((DATA_DIR / f"effis_ba_{META['effis_id']}.json").read_text())
    points = []
    for polygon in effis["shape"]["coordinates"]:
        ring = np.asarray(polygon[0])[::step]
        x, y = lonlat_to_km(ring[:, 0], ring[:, 1])
        points.append(np.column_stack([x, y]))
    return np.concatenate(points)


def sanity_check() -> None:
    truth_ha = TRUTH_MASK.sum() * CELL_HA
    assert abs(truth_ha - EFFIS_BURNED_HA) / EFFIS_BURNED_HA < 0.03, truth_ha
    assert ELEVATION_M.min() > -5.0 and ELEVATION_M.max() < 1000.0
    assert FUEL_CLASS.shape == (NY, NX) == TRUTH_MASK.shape == ELEVATION_M.shape
    assert FUEL_CLASS[IGNITION_IY, IGNITION_IX] > 0, "ignition cell must be burnable"
    # The perimeter must lie fully inside the domain (with margin for spread).
    perim = truth_perimeter_km(step=1)
    assert perim[:, 0].min() > 1.0 and perim[:, 0].max() < (NX - 3) * CELL_KM
    assert perim[:, 1].min() > 1.0 and perim[:, 1].max() < (NY - 3) * CELL_KM
    # Cross-check land cover against the EFFIS per-class breakdown: EFFIS
    # reports mostly sclerophyllous/transitional shrub + conifer inside the
    # scar; our fuel map must be dominated by shrub + forest there as well.
    woody = np.isin(FUEL_CLASS[TRUTH_MASK], (2, 3)).mean()
    assert woody > 0.6, f"woody fraction inside scar = {woody:.2f}"


if __name__ == "__main__":
    sanity_check()
    truth_ha = TRUTH_MASK.sum() * CELL_HA
    print(f"grid {NX} x {NY} cells of {CELL_M:.0f} m ({NX * CELL_KM:.1f} x {NY * CELL_KM:.1f} km)")
    print(f"elevation {ELEVATION_M.min():.0f}..{ELEVATION_M.max():.0f} m")
    print(
        f"truth scar {truth_ha:.0f} ha (EFFIS {EFFIS_BURNED_HA} ha, official {OFFICIAL_BURNED_HA} ha)"
    )
    counts = np.bincount(FUEL_CLASS.ravel(), minlength=4)
    print(f"fuel cells none/grass/shrub/forest: {counts.tolist()}")
    print(f"ignition cell ({IGNITION_IX}, {IGNITION_IY})")
    print("sanity checks passed")
