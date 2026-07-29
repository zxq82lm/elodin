"""One-time data preparation for the wildfire-aude example.

Fetches open data and derives the simulation grids vendored as `grids.npz`
(+ `meta.json`). The simulation itself never touches the network: it only
loads the derived files through `reference.py`.

Sources (all free and open):
- Burnt-area perimeter: EFFIS (European Forest Fire Information System,
  Copernicus), burnt area id 276934 — the 2025 Corbieres massif fire
  (Aude, France; started near Ribaute on 2025-08-05, ~11 391 ha mapped).
  https://api.effis.emergency.copernicus.eu/rest/2/burntareas/current/276934/
  The verbatim response is vendored as `effis_ba_276934.json`.
- Elevation: Copernicus DEM GLO-30 (ESA), public COGs on AWS S3.
  https://copernicus-dem-30m.s3.amazonaws.com/
- Land cover: ESA WorldCover 10 m v200 (2021), public COGs on AWS S3.
  https://esa-worldcover.s3.eu-central-1.amazonaws.com/

Usage (network required, run from the repo root inside `nix develop`):
    uv run --with rasterio --with matplotlib examples/wildfire-aude/data/prep_data.py
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent
EFFIS_ID = 276934
EFFIS_URL = (
    f"https://api.effis.emergency.copernicus.eu/rest/2/burntareas/current/{EFFIS_ID}/?format=json"
)
DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif"
)
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N42E000_Map.tif"
)

# Simulation domain (WGS84). The EFFIS bbox is lon 2.6209..2.9087,
# lat 42.9775..43.1122; margins leave room for ignition jitter and wind
# direction dispersion in Monte Carlo campaigns.
LON_MIN, LON_MAX = 2.55, 2.97
LAT_MIN, LAT_MAX = 42.95, 43.14
CELL_M = 300.0

# Documented ignition area: roadside between Ribaute and Lagrasse,
# afternoon of 2025-08-05 (Prefecture of Aude / press reports).
IGNITION_LON, IGNITION_LAT = 2.615, 43.085

# WorldCover class -> fuel class used by the simulation.
# Fuel classes: 0 = non-burnable, 1 = grass/agriculture, 2 = shrub
# (garrigue), 3 = forest.
WORLDCOVER_TO_FUEL = {
    10: 3,  # tree cover
    20: 2,  # shrubland
    30: 1,  # grassland
    40: 1,  # cropland
    50: 0,  # built-up
    60: 0,  # bare / sparse vegetation
    70: 0,  # snow and ice
    80: 0,  # permanent water bodies
    90: 1,  # herbaceous wetland
    95: 0,  # mangroves
    100: 1,  # moss and lichen
}


def local_meters_per_degree(lat_deg: float) -> tuple[float, float]:
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_deg))
    return m_per_deg_lon, m_per_deg_lat


def grid_geometry() -> dict:
    lat_mid = 0.5 * (LAT_MIN + LAT_MAX)
    m_lon, m_lat = local_meters_per_degree(lat_mid)
    width_m = (LON_MAX - LON_MIN) * m_lon
    height_m = (LAT_MAX - LAT_MIN) * m_lat
    nx = int(round(width_m / CELL_M))
    ny = int(round(height_m / CELL_M))
    return {
        "lon_min": LON_MIN,
        "lat_min": LAT_MIN,
        "lon_max": LON_MAX,
        "lat_max": LAT_MAX,
        "m_per_deg_lon": m_lon,
        "m_per_deg_lat": m_lat,
        "cell_m": CELL_M,
        "nx": nx,
        "ny": ny,
    }


def cell_centers_lonlat(geom: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center coordinates, row 0 = south, col 0 = west."""
    xs = (np.arange(geom["nx"]) + 0.5) * geom["cell_m"] / geom["m_per_deg_lon"] + geom["lon_min"]
    ys = (np.arange(geom["ny"]) + 0.5) * geom["cell_m"] / geom["m_per_deg_lat"] + geom["lat_min"]
    return np.meshgrid(xs, ys)


def fetch_effis() -> dict:
    path = DATA_DIR / f"effis_ba_{EFFIS_ID}.json"
    if not path.exists():
        print(f"downloading {EFFIS_URL}")
        with urllib.request.urlopen(EFFIS_URL, timeout=60) as resp:
            path.write_bytes(resp.read())
    return json.loads(path.read_text())


def read_dem(geom: dict) -> np.ndarray:
    """Sample Copernicus GLO-30 elevation (m) at every cell center."""
    import rasterio
    from rasterio.transform import rowcol

    lon_grid, lat_grid = cell_centers_lonlat(geom)
    elevation = np.full(lon_grid.shape, np.nan, dtype=np.float32)
    # 1x1 degree tiles; the domain spans the 43N boundary.
    for lat_tile in sorted({math.floor(lat) for lat in (LAT_MIN, LAT_MAX)}):
        url = DEM_URL.format(lat=f"N{lat_tile}", lon="E002")
        print(f"reading DEM window from {url}")
        with rasterio.open(url) as src:
            sel = (lat_grid >= lat_tile) & (lat_grid < lat_tile + 1)
            if not sel.any():
                continue
            rows, cols = rowcol(src.transform, lon_grid[sel], lat_grid[sel])
            band = src.read(1)
            elevation[sel] = band[np.asarray(rows), np.asarray(cols)]
    assert not np.isnan(elevation).any(), "DEM sampling left gaps"
    return elevation


def read_fuel(geom: dict) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate WorldCover 10 m classes to (fuel_class, burnable_frac)."""
    import rasterio
    from rasterio.windows import from_bounds

    print(f"reading WorldCover window from {WORLDCOVER_URL}")
    with rasterio.open(WORLDCOVER_URL) as src:
        window = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
        cover = src.read(1, window=window)
        transform = src.window_transform(window)

    ny, nx = geom["ny"], geom["nx"]
    # Simulation-grid cell index of each WorldCover pixel (pixel centers).
    px_lon = transform.c + transform.a * (np.arange(cover.shape[1]) + 0.5)
    px_lat = transform.f + transform.e * (np.arange(cover.shape[0]) + 0.5)
    ix = ((px_lon - geom["lon_min"]) * geom["m_per_deg_lon"] / geom["cell_m"]).astype(int)
    iy = ((px_lat - geom["lat_min"]) * geom["m_per_deg_lat"] / geom["cell_m"]).astype(int)
    col_ok = (ix >= 0) & (ix < nx)
    row_ok = (iy >= 0) & (iy < ny)
    cover = cover[np.ix_(row_ok, col_ok)]
    cell = iy[row_ok][:, None] * nx + ix[col_ok][None, :]

    fuel_lut = np.zeros(256, dtype=np.uint8)
    for wc_class, fuel in WORLDCOVER_TO_FUEL.items():
        fuel_lut[wc_class] = fuel
    fuel_px = fuel_lut[cover]

    counts = np.zeros((ny * nx, 4), dtype=np.int64)
    for fuel in range(4):
        counts[:, fuel] = np.bincount(cell.ravel()[fuel_px.ravel() == fuel], minlength=ny * nx)
    total = counts.sum(axis=1)
    assert (total > 0).all(), "empty simulation cells during aggregation"
    fuel_class = counts[:, 1:].argmax(axis=1) + 1
    fuel_class[counts[:, 1:].sum(axis=1) == 0] = 0
    burnable_frac = counts[:, 1:].sum(axis=1) / total
    return (
        fuel_class.reshape(ny, nx).astype(np.uint8),
        burnable_frac.reshape(ny, nx).astype(np.float32),
    )


def rasterize_truth(effis: dict, geom: dict) -> np.ndarray:
    """Rasterize the EFFIS burnt-area MultiPolygon onto the grid."""
    from matplotlib.path import Path as MplPath

    lon_grid, lat_grid = cell_centers_lonlat(geom)
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    mask = np.zeros(pts.shape[0], dtype=bool)
    for polygon in effis["shape"]["coordinates"]:
        outer = MplPath(np.asarray(polygon[0]))
        inside = outer.contains_points(pts)
        for hole in polygon[1:]:
            inside &= ~MplPath(np.asarray(hole)).contains_points(pts)
        mask |= inside
    return mask.reshape(geom["ny"], geom["nx"])


def main() -> None:
    geom = grid_geometry()
    print(f"grid: {geom['nx']} x {geom['ny']} cells of {geom['cell_m']:.0f} m")

    effis = fetch_effis()
    elevation = read_dem(geom)
    fuel_class, burnable_frac = read_fuel(geom)
    truth = rasterize_truth(effis, geom)

    ix = int((IGNITION_LON - geom["lon_min"]) * geom["m_per_deg_lon"] / geom["cell_m"])
    iy = int((IGNITION_LAT - geom["lat_min"]) * geom["m_per_deg_lat"] / geom["cell_m"])
    assert 0 <= ix < geom["nx"] and 0 <= iy < geom["ny"]

    np.savez_compressed(
        DATA_DIR / "grids.npz",
        elevation=elevation,
        fuel_class=fuel_class,
        burnable_frac=burnable_frac,
        truth=truth.astype(np.uint8),
    )
    meta = dict(
        geom,
        ignition_ix=ix,
        ignition_iy=iy,
        ignition_lon=IGNITION_LON,
        ignition_lat=IGNITION_LAT,
        effis_id=EFFIS_ID,
        effis_area_ha=effis["area_ha"],
        effis_firedate=effis["firedate"],
        sources=dict(effis=EFFIS_URL, dem=DEM_URL, worldcover=WORLDCOVER_URL),
    )
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    truth_ha = truth.sum() * (geom["cell_m"] / 100.0) ** 2
    print(f"truth mask: {truth.sum()} cells = {truth_ha:.0f} ha (EFFIS: {effis['area_ha']} ha)")
    print(f"elevation: {elevation.min():.0f}..{elevation.max():.0f} m")
    print(f"fuel classes (cells): {np.bincount(fuel_class.ravel(), minlength=4).tolist()}")
    print(f"ignition cell: ({ix}, {iy}), fuel={fuel_class[iy, ix]}")


if __name__ == "__main__":
    main()
