"""
power_albedo.py — NASA POWER daily surface albedo fetcher with local JSON cache.

Tier A albedo for MSC ESH simulation.  Queries the POWER Daily API for
ALLSKY_SRF_ALB (or CLRSKY_SRF_ALB) at the ISS sub-satellite lat/lon and
returns a per-timestep rho_eff Series that can be merged into the main df.

Usage
-----
    from power_albedo import build_power_rho_series
    rho_series = build_power_rho_series(lla_df, cache_dir="data/cache")
    # rho_series: DataFrame with columns ["t", "rho_eff"]

API reference
-------------
    https://power.larc.nasa.gov/api/temporal/daily/point
    parameters: ALLSKY_SRF_ALB  (all-sky surface albedo, dimensionless)
                CLRSKY_SRF_ALB  (clear-sky surface albedo, dimensionless)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POWER_API_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
GRID_RES_DEG   = 0.5   # snap lat/lon to nearest 0.5° POWER grid centre
POWER_FILL     = -999   # POWER uses -999 for missing / fill values


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def _snap(value: float, res: float = GRID_RES_DEG) -> float:
    """Snap a coordinate to the nearest POWER grid centre."""
    return round(round(value / res) * res, 4)


def _cache_path(cache_dir: Path, lat: float, lon: float, date_str: str) -> Path:
    return cache_dir / f"power_{lat:.2f}_{lon:.2f}_{date_str}.json"


# ---------------------------------------------------------------------------
# Single-cell fetch
# ---------------------------------------------------------------------------
def fetch_one(
    lat: float,
    lon: float,
    date_str: str,
    cache_dir: Path,
    parameter: str = "ALLSKY_SRF_ALB",
    retries: int = 3,
) -> float:
    """
    Fetch one daily POWER albedo value for a snapped (lat, lon) grid cell.

    Results are cached as JSON files in *cache_dir* so repeated simulation
    runs never re-query the same cell.

    Parameters
    ----------
    lat, lon   : float — coordinates already snapped to POWER grid.
    date_str   : str   — date in YYYYMMDD format.
    cache_dir  : Path  — directory for JSON cache files (created if absent).
    parameter  : str   — POWER parameter name.
    retries    : int   — max retry attempts with exponential back-off.

    Returns
    -------
    float — dimensionless albedo value, or np.nan on failure / fill.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf = _cache_path(cache_dir, lat, lon, date_str)

    if cf.exists():
        with open(cf) as f:
            return float(json.load(f).get("value", np.nan))

    url = (
        f"{POWER_API_BASE}"
        f"?parameters={parameter}"
        f"&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={date_str}"
        f"&end={date_str}"
        f"&format=JSON"
        f"&time-standard=UTC"
    )

    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            raw = payload["properties"]["parameter"][parameter][date_str]
            val = float(raw)
            if val <= POWER_FILL + 1:   # treat -999 and below as fill
                val = np.nan
            with open(cf, "w") as f:
                json.dump(
                    {"value": val, "parameter": parameter,
                     "lat": lat, "lon": lon, "date": date_str},
                    f,
                )
            return val
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)   # 1 s, 2 s back-off

    print(f"  [POWER] fetch failed ({lat:.2f}, {lon:.2f}, {date_str}): {last_exc}")
    return np.nan


# ---------------------------------------------------------------------------
# Series builder
# ---------------------------------------------------------------------------
def build_power_rho_series(
    lla_df: pd.DataFrame,
    cache_dir: str = "data/cache",
    parameter: str = "ALLSKY_SRF_ALB",
    fallback: float = 0.27,
    query_interval_minutes: int = 1440,
) -> pd.DataFrame:
    """
    Build a per-timestep rho_eff series from NASA POWER surface albedo.

    For each timestep the sub-satellite lat/lon is snapped to the nearest
    0.5° POWER grid cell. POWER fetches are made at a reduced cadence
    controlled by ``query_interval_minutes`` and then applied to all timesteps
    in the same day/slot.

    Parameters
    ----------
    lla_df    : DataFrame with columns 't' (datetime64), 'Lat (deg)', 'Lon (deg)'.
    cache_dir : Local directory for JSON cache files.
    parameter : POWER parameter — 'ALLSKY_SRF_ALB' or 'CLRSKY_SRF_ALB'.
    fallback  : Value used when POWER returns NaN or a fill value.
    query_interval_minutes : POWER query cadence per UTC day.
        Default 1440 => one POWER call/day.
        Example: 360 => at most 4 POWER calls/day (00:00, 06:00, 12:00, 18:00 slots).
        Must be >= 1.

    Returns
    -------
    DataFrame with columns ['t', 'rho_eff'].
    """
    if query_interval_minutes < 1:
        raise ValueError("query_interval_minutes must be >= 1")

    cache_path = Path(cache_dir)
    df = lla_df.copy().sort_values("t").reset_index(drop=True)

    # Snap sub-satellite track to nearest POWER grid cells
    df["lat_s"]    = df["Lat (deg)"].apply(_snap)
    df["lon_s"]    = df["Lon (deg)"].apply(_snap)
    df["date_str"] = df["t"].dt.strftime("%Y%m%d")

    # Build per-day query slot so API calls are bounded for long simulations
    day_floor = df["t"].dt.floor("D")
    minutes_since_day_start = (
        (df["t"] - day_floor).dt.total_seconds() // 60
    ).astype(int)
    df["query_slot"] = (minutes_since_day_start // query_interval_minutes).astype(int)

    # One representative row per (day, slot)
    query_points = (
        df[["date_str", "query_slot", "lat_s", "lon_s"]]
        .drop_duplicates(subset=["date_str", "query_slot"], keep="first")
        .reset_index(drop=True)
    )

    total = len(query_points)
    print(
        f"[POWER] Querying {total} day/slot points "
        f"(parameter={parameter}, interval={query_interval_minutes} min) ..."
    )

    # Fetch / cache one value per unique (lat, lon, day) key
    cell_lookup: dict = {}
    slot_lookup: dict = {}
    for idx, row in enumerate(query_points.itertuples(index=False), 1):
        cell_key = (row.lat_s, row.lon_s, row.date_str)
        if cell_key not in cell_lookup:
            cell_lookup[cell_key] = fetch_one(
                row.lat_s, row.lon_s, row.date_str, cache_path, parameter
            )
        slot_key = (row.date_str, row.query_slot)
        val = cell_lookup[cell_key]
        slot_lookup[slot_key] = val if np.isfinite(val) else fallback
        if idx % 50 == 0 or idx == total:
            print(f"  {idx}/{total} slots done")

    # Map slot albedo back to every timestep
    df["rho_eff"] = [
        slot_lookup.get((d, s), fallback)
        for d, s in zip(df["date_str"], df["query_slot"])
    ]

    valid_hits = sum(1 for v in cell_lookup.values() if np.isfinite(v))
    print(
        f"[POWER] Done — {valid_hits}/{total} slots had valid data "
        f"({total - valid_hits} used fallback={fallback:.3f})"
    )

    if valid_hits == 0:
        d0 = df["date_str"].min()
        d1 = df["date_str"].max()
        print(
            "[POWER] WARNING — all values were missing/fill, so fallback was used "
            f"for every slot. Requested date range: {d0} to {d1}. "
            "This often indicates parameter publication lag for recent dates."
        )

    return df[["t", "rho_eff"]].reset_index(drop=True)
