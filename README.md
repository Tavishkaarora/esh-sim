# MSC ESH Simulation — Aegis

**Equivalent Sun Hours (ESH) simulation for the MISSE Science Carrier (MSC) on the International Space Station**

This repository contains a production-ready pipeline for computing the cumulative solar irradiance (expressed as Equivalent Sun Hours) incident on each face of the MSC instrument package across any simulation period. It ingests telemetry exports from **Systems Tool Kit (STK)**, applies eclipse, structural-blocking, and operational-exposure masks, and produces per-face ESH time series consumed by downstream sensor-degradation models.

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [What Is ESH and Why It Matters](#2-what-is-esh-and-why-it-matters)
3. [System Architecture and Data Flow](#3-system-architecture-and-data-flow)
4. [File-by-File Reference](#4-file-by-file-reference)
   - [power_albedo.py](#41-power_albedopy)
   - [streamlit_app.py](#42-streamlit_apppy)
   - [esh_final_simulation.ipynb](#43-esh_final_simulationipynb)
   - [esh_prototype.ipynb](#44-esh_prototypeipynb)
   - [data/ — STK Exports](#45-data--stk-exports)
   - [outputs/](#46-outputs)
5. [Running the Streamlit UI](#5-running-the-streamlit-ui)
6. [Running the Jupyter Notebooks](#6-running-the-jupyter-notebooks)
7. [Generating New STK Reports (Changing the Simulation Period)](#7-generating-new-stk-reports-changing-the-simulation-period)
8. [Key Configurable Parameters](#8-key-configurable-parameters)
9. [Albedo Options](#9-albedo-options)
10. [Understanding the Three Masking Layers](#10-understanding-the-three-masking-layers)
11. [Output Files and Downstream Use](#11-output-files-and-downstream-use)
12. [Scaling the System](#12-scaling-the-system)
13. [Known Limitations and Future Work](#13-known-limitations-and-future-work)
14. [Environment Setup](#14-environment-setup)

---

## 1. Repository Layout

```
esh-estimate-simulation-aegis/
├── .gitignore                          # Excludes large STK CSVs, outputs, .venv
└── simulation/
    └── msc-esh-6-month-sim/            # Primary simulation package
        ├── data/                        # STK CSV exports (inputs)
        │   ├── ISS_Model_J2000_Position_Velocity.csv
        │   ├── ISS_Model_Sun_Vector_J2000.csv
        │   ├── ISS_Model_LLA_Position.csv
        │   ├── ISS_Model_Lighting_Times.csv
        │   ├── MSC_Sun_Sensor_Az_El_Mask.csv
        │   └── cache/                   # NASA POWER API cached JSON responses
        │       └── power_<lat>_<lon>_<date>.json
        ├── outputs/                     # Generated plots and result CSVs
        │   ├── msc_esh_6month_irradiance.csv   ← main simulation output
        │   ├── fig_azel_diagnostic.png
        │   ├── fig_cumulative_esh_all.png
        │   ├── fig_cumulative_esh_faces.png
        │   ├── fig_direct_irradiance.png
        │   ├── fig_exposure_mask.png
        │   ├── fig_structural_blocking.png
        │   ├── fig_sun_eclipse.png
        │   ├── fig_total_irradiance.png
        │   └── fig_per_orbit_flux.png
        ├── esh_final_simulation.ipynb   # Production pipeline (6-month)
        ├── esh_prototype.ipynb          # 1-day prototype / validation run
        ├── power_albedo.py              # NASA POWER API module
        ├── streamlit_app.py             # Interactive web UI
        └── prototype_face_irradiance_1day.csv  # 1-day test output
```

---

## 2. What Is ESH and Why It Matters

**Equivalent Sun Hours (ESH)** is a dimensionless measure of cumulative solar energy exposure on a surface, normalised to the solar constant (AM0 = 1361 W/m²):

```
ESH = ∫ I_effective(t) dt  /  (AM0 × 3600)
```

where `I_effective` is the masked irradiance (W/m²) at each timestep and the integral is in seconds, producing a result in **sun-hours**. One ESH = one hour of full, unobstructed AM0 illumination.

For space-based instruments like the MSC sensors, ESH drives:
- UV-induced degradation of optical coatings and detector materials
- Thermal cycling fatigue (indirect)
- Calibration correction factors in sensor response models

The simulation produces a **continuous ESH time series** for each of the six faces of the MSC package (+R, -R, +T, -T, +H, -H in LVLH frame coordinates), which downstream models convert to sensor-specific degradation estimates.

---

## 3. System Architecture and Data Flow

```
STK Scenario (ISS orbit propagation)
         │
         │  Export reports (CSV, 1-minute cadence)
         ▼
┌─────────────────────────────────────────────────┐
│               data/ directory                   │
│  J2000 Position/Velocity  ←─ inertial state     │
│  J2000 Sun Vector         ←─ Sun direction      │
│  LLA Position             ←─ sub-satellite lat/lon │
│  Lighting Times           ←─ eclipse intervals  │
│  Az/El Mask               ←─ structural block   │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│           SIMULATION ENGINE                     │
│  (streamlit_app.py  OR  esh_final_simulation.ipynb) │
│                                                 │
│  1. Parse & merge STK CSVs                      │
│  2. Build LVLH reference frame                  │
│  3. Compute Sun direction in LVLH               │
│  4. Fetch albedo (NASA POWER API / constant)    │
│  5. Compute direct + albedo irradiance per face │
│  6. Apply eclipse mask (factor: 1.0/0.5/0.0)   │
│  7. Apply structural mask (factor: 0 or 1)      │
│  8. Apply exposure-window mask (factor: 0 or 1) │
│  9. Integrate → ESH per face                    │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              outputs/                           │
│  msc_esh_6month_irradiance.csv ← main output   │
│  Diagnostic PNG plots                           │
└─────────────────────────────────────────────────┘
         │
         ▼
   Downstream sensor degradation models
   (sensor_channel_split.ipynb, etc.)
```

**The LVLH Frame:** All geometry is computed in the Local Vertical Local Horizontal frame:
- **+R (Radial):** Points away from Earth centre (zenith)
- **-R (Nadir):** Points toward Earth
- **+T (Along-track):** Velocity direction
- **-T (Anti-velocity):** Trailing face
- **+H (Orbit Normal):** Perpendicular to orbital plane
- **-H (Anti-normal):** Opposite orbit normal

---

## 4. File-by-File Reference

### 4.1 `power_albedo.py`

**Role:** Self-contained module for fetching and caching Earth surface albedo data from the NASA POWER REST API.

**Key functions:**

| Function | Purpose |
|---|---|
| `_snap(value, res=0.5)` | Round a coordinate to the nearest 0.5° POWER grid centre |
| `_cache_path(cache_dir, lat, lon, date_str)` | Build the JSON cache file path for a given lat/lon/date |
| `fetch_one(lat, lon, date_str, cache_dir, parameter, retries)` | Fetch a single day's albedo value. Checks disk cache first; queries API on miss; retries with exponential backoff. Returns `np.nan` on failure. |
| `build_power_rho_series(lla_df, cache_dir, parameter, fallback, query_interval_minutes)` | **Main entry point.** Takes ISS LLA DataFrame, snaps each position to the POWER grid, reduces API calls to one per `query_interval_minutes` (default = 1440 min = daily), returns `DataFrame[t, rho_eff]` with nan-filled values replaced by `fallback`. |

**NASA POWER API endpoint:**
```
https://power.larc.nasa.gov/api/temporal/daily/point
  ?parameters=ALLSKY_SRF_ALB
  &community=RE
  &longitude=<lon>
  &latitude=<lat>
  &start=<YYYYMMDD>
  &end=<YYYYMMDD>
  &format=JSON
```

**Cache format** (`data/cache/power_<lat>_<lon>_<date>.json`):
```json
{"value": 0.08, "parameter": "ALLSKY_SRF_ALB", "lat": 27.0, "lon": -86.5, "date": "20200811"}
```

---

### 4.2 `streamlit_app.py`

**Role:** Interactive browser-based web UI wrapping the full simulation engine. Also serves as a standalone Python module (the simulation logic can be imported directly).

**Sidebar configuration options:**

| Control | Default | Description |
|---|---|---|
| Data directory | `data/` | Folder containing all 5 STK CSV files |
| Primary face | `+H` | Which face to feature in the main irradiance plot |
| Albedo mode | `power` | `power` (NASA POWER API) or `constant` |
| Constant albedo | 0.27 | Used when mode is `constant`, or as POWER fallback |
| POWER parameter | `ALLSKY_SRF_ALB` | `ALLSKY_SRF_ALB` (cloudy+clear) or `CLRSKY_SRF_ALB` (clear only) |
| POWER cache dir | `data/cache` | Where to read/write cached POWER JSON files |
| Exposure windows | *(blank)* | Operational open intervals (one per line, start → stop) |

**Core simulation function:** `run_simulation(config: dict, log: callable) → dict`

Returns a results dictionary containing:
- `df` — full timestep-by-timestep DataFrame with all computed columns
- `esh_summary` — dict of total ESH per face
- `export_df` — downstream-compatible DataFrame (same schema as the main CSV output)
- `diag` — diagnostic dict (albedo stats, eclipse event counts, etc.)

**Plot tabs generated:**
1. **Primary face** — 3-panel: eclipse factor timeline, direct + albedo irradiance, cumulative ESH
2. **All faces** — Cumulative ESH for all 6 faces on one axes
3. **Albedo** — Effective albedo along ISS groundtrack

**Download section:** Two CSV downloads — face-specific export and full simulation DataFrame.

**Face normal vectors (LVLH, generic box model):**
```python
"+R": [ 1,  0,  0]   # Zenith
"-R": [-1,  0,  0]   # Nadir
"+T": [ 0,  1,  0]   # Along-track (velocity)
"-T": [ 0, -1,  0]   # Anti-velocity
"+H": [ 0,  0,  1]   # Orbit normal
"-H": [ 0,  0, -1]   # Anti-orbit normal
```

---

### 4.3 `esh_final_simulation.ipynb`

**Role:** The complete, annotated reference implementation for a 6-month ESH simulation. Every computational step is explained with Markdown cells. This is the authoritative source of truth for the pipeline physics.

**Notebook sections:**
1. **Imports and constants** — Physical constants, file paths
2. **User configuration block** — Edit `PRIMARY_FACE`, `EXPOSURE_WINDOWS`, `ALBEDO_MODE` here
3. **STK data loading** — Parse and merge all 5 CSV exports into one aligned DataFrame
4. **LVLH frame construction** — Compute R̂, T̂, Ĥ unit vectors from position and velocity
5. **Sun direction in LVLH** — Project sun vector onto LVLH basis
6. **Albedo fetching** — NASA POWER or constant; `rho_eff` time series
7. **Irradiance computation** — Direct and albedo irradiance per face per timestep
8. **Eclipse masking** — Parse lighting times, assign per-step eclipse factor
9. **Structural masking** — Az/El containment tests on exclusion zone polygons
10. **Exposure masking** — Merge user-defined operational windows into binary mask
11. **Final masked irradiance** — Combine all three masks
12. **ESH integration** — Cumulative sum → ESH per face
13. **Plots** — Diagnostic figures written to `outputs/`
14. **CSV export** — Write `outputs/msc_esh_6month_irradiance.csv`

---

### 4.4 `esh_prototype.ipynb`

**Role:** Single-day (4–5 Mar 2026) test case used to validate the pipeline end-to-end before running the full 6-month simulation. Identical structure to the production notebook.

**Purpose of this notebook:**
- Quick validation that STK data parses correctly
- Sanity-check irradiance magnitudes and eclipse patterns
- Diagnostics on structural blocking (note: that date has a high beta angle β ≈ 42°, so 0% structural blocking is physically correct — the Sun stays above all exclusion zone elevation ceilings at ~35.84°)

---

### 4.5 `data/` — STK Exports

All five files below must be present for the simulation to run. Each is a CSV exported directly from an STK report.

#### `ISS_Model_J2000_Position_Velocity.csv`
**STK Report Type:** Satellite state vector (J2000 inertial frame)

| Column | Units | Description |
|---|---|---|
| `Time (UTCG)` | string | Timestamp: `DD Mon YYYY HH:MM:SS.mmm` |
| `x (km)` | km | ISS X position in J2000 |
| `y (km)` | km | ISS Y position in J2000 |
| `z (km)` | km | ISS Z position in J2000 |
| `vx (km/sec)` | km/s | ISS X velocity in J2000 |
| `vy (km/sec)` | km/s | ISS Y velocity in J2000 |
| `vz (km/sec)` | km/s | ISS Z velocity in J2000 |

Example row:
```
1 Jul 2020 18:00:00.000,2246.075312,5350.587290,3527.528641,-4.085381,4.683376,-4.486543
```

#### `ISS_Model_Sun_Vector_J2000.csv`
**STK Report Type:** Sun position in J2000 frame (from ISS)

| Column | Units | Description |
|---|---|---|
| `Time (UTCG)` | string | Timestamp |
| `x (km)` | km | Sun X position in J2000 |
| `y (km)` | km | Sun Y position in J2000 |
| `z (km)` | km | Sun Z position in J2000 |

The simulation computes `sun_dir = (sun_pos - iss_pos) / |...|` as a unit vector.

#### `ISS_Model_LLA_Position.csv`
**STK Report Type:** ISS geodetic (lat/lon/alt) position

| Column | Units | Description |
|---|---|---|
| `Time (UTCG)` | string | Timestamp |
| `Lat (deg)` | degrees | Geodetic latitude of ISS sub-satellite point |
| `Lon (deg)` | degrees | Geodetic longitude |
| `Alt (km)` | km | Altitude above WGS84 ellipsoid |
| `Lat Rate (deg/sec)` | deg/s | Rate of change |
| `Lon Rate (deg/sec)` | deg/s | Rate of change |
| `Alt Rate (km/sec)` | km/s | Rate of change |

Used only for the NASA POWER albedo API query (lat/lon snapped to 0.5° grid).

#### `ISS_Model_Lighting_Times.csv`
**STK Report Type:** Access intervals (lighting condition)

Three sequential tables in one file, separated by STK "Global Statistics" blocks:

| Table | Lighting state | Typical duration |
|---|---|---|
| Table 0 | **Sunlight** | ~56 minutes per interval |
| Table 1 | **Penumbra** | ~12 seconds per interval |
| Table 2 | **Umbra** | ~34 minutes per interval |

Each table has columns: `Start Time (UTCG)`, `Stop Time (UTCG)`, `Duration (sec)`.

#### `MSC_Sun_Sensor_Az_El_Mask.csv`
**STK Report Type:** Custom Az/El constraint report (MSC sensor field-of-view mask)

Contains polygon boundary points for:
- **6 Exclusion Zones** — regions of the Az/El sphere blocked by ISS structure (solar arrays, truss, body)
- **57 Inclusion Zones** — gaps within exclusion zones (windows in the structure)

Each zone is a named block of rows:
```
"Exclusion Zone 3"
Point, Azimuth (deg), Elevation (deg)
1, -44.775, 34.431
2, -40.250, 34.364
...
```

The simulation uses `matplotlib.path.Path.contains_points()` to test whether the Sun's Az/El position falls inside an exclusion zone but outside any inclusion zone within it, flagging the timestep as structurally blocked.

**Az/El convention:**
- Azimuth: `atan2(s_T, s_R)` — measured in the orbital plane, zero toward radially outward
- Elevation: `arcsin(s_H)` — angle above the orbital plane

---

### 4.6 `outputs/`

| File | Description |
|---|---|
| `msc_esh_6month_irradiance.csv` | **Main output.** Full time series of masked irradiance + ESH per face (see schema in §11) |
| `fig_azel_diagnostic.png` | Sun Az/El track overlaid on exclusion zone polygons |
| `fig_cumulative_esh_all.png` | Cumulative ESH for all 6 faces over simulation period |
| `fig_cumulative_esh_faces.png` | Per-face ESH breakdown (bar chart or multi-line) |
| `fig_direct_irradiance.png` | Direct solar irradiance comparison across faces |
| `fig_exposure_mask.png` | Timeline showing operational exposure open/closed windows |
| `fig_structural_blocking.png` | Timeline of structural blocking events |
| `fig_sun_eclipse.png` | Sun elevation angle + eclipse factor together |
| `fig_total_irradiance.png` | Total irradiance (direct + albedo) for primary face |
| `fig_per_orbit_flux.png` | Single-orbit irradiance profile |

---

## 5. Running the Streamlit UI

### Prerequisites

Activate the project virtual environment:
```bash
cd /Users/tavishka/esh-estimate-simulation-aegis
source .venv/bin/activate
```

### Launch

```bash
cd simulation/msc-esh-6-month-sim
streamlit run streamlit_app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`). Open it in your browser.

### Workflow in the UI

1. **Data directory** — Confirm it points to the folder containing your 5 STK CSVs. Default is `data/` (relative to the app's working directory).

2. **Primary face** — Select the MSC face you want featured in the main irradiance plot. All 6 faces are always simulated; this just controls the default detail view.

3. **Albedo mode:**
   - `power` — Fetches real daily albedo from NASA POWER. First run requires internet; subsequent runs use cache (fast).
   - `constant` — Uses the fixed albedo value you enter (default 0.27). Faster, no internet required.

4. **Exposure windows (optional)** — If the MSC has defined operational windows (periods when the instrument shutter/aperture is open), enter them one per line in the format:
   ```
   04 Mar 2026 10:54:00.000 → 04 Mar 2026 11:54:00.000
   05 Mar 2026 09:00:00.000 → 05 Mar 2026 10:30:00.000
   ```
   Leave blank to treat the entire simulation period as "always open."

5. **Click "Run Simulation"** — The progress log in the main panel shows each pipeline step. A 6-month simulation at 1-minute cadence (~265,000 timesteps) takes ~30–60 seconds depending on albedo mode.

6. **Review results:**
   - ESH summary table at the top
   - Tabbed plot panels below
   - Simulation info (duration, albedo stats, eclipse counts)

7. **Download results** using the CSV export buttons in the "Downloads" section.

### Using the App for Different Simulation Periods

After generating new STK reports (see §7), simply:
1. Place the new CSV files in `data/` (overwriting or in a new subfolder)
2. Update the **Data directory** path in the sidebar
3. Click **Run Simulation**

No code changes required.

---

## 6. Running the Jupyter Notebooks

### Activate environment and launch Jupyter

```bash
cd /Users/tavishka/esh-estimate-simulation-aegis
source .venv/bin/activate
jupyter lab
```

Navigate to `simulation/msc-esh-6-month-sim/esh_final_simulation.ipynb`.

### User configuration block

Near the top of the notebook, find the cell labelled **"User Configuration"**. Edit these variables before running:

```python
# ── Which MSC face to treat as primary ──────────────────────────────────────
PRIMARY_FACE = "+H"   # Options: "+R", "-R", "+T", "-T", "+H", "-H"

# ── Operational exposure windows ─────────────────────────────────────────────
# List of (start, end) tuples in STK timestamp format. Leave empty for always-open.
EXPOSURE_WINDOWS = [
    # ("04 Mar 2026 10:54:00.000", "04 Mar 2026 11:54:00.000"),
]

# ── Albedo model ─────────────────────────────────────────────────────────────
ALBEDO_MODE = "power"     # "power" | "constant"
ALBEDO_CONSTANT = 0.27    # Used when mode is "constant" or as POWER fallback
POWER_PARAM = "ALLSKY_SRF_ALB"   # or "CLRSKY_SRF_ALB"
```

Then run all cells: **Kernel → Restart & Run All**.

### Prototype notebook

`esh_prototype.ipynb` is structured identically. Use it when you want to:
- Validate a new STK export on a single day before committing to a full run
- Debug parsing or geometry issues quickly
- Test a new exposure window definition

---

## 7. Generating New STK Reports (Changing the Simulation Period)

To simulate a different time range or a different ISS scenario, you must regenerate all five STK CSV reports. Below are the exact steps for each report.

### 7.1 Prerequisites in STK

- Open your STK scenario containing the **ISS satellite object** and the **MSC sensor object**
- Set the scenario analysis interval to cover your desired simulation period
  - **Analysis → Time Period** (or the scenario properties) → set Start and Stop times
  - Use UTC time. STK exports will use the format `DD Mon YYYY HH:MM:SS.mmm`

### 7.2 Report 1 — ISS State Vector (J2000 Position & Velocity)

**Target object:** ISS satellite

1. Right-click the ISS satellite → **Report & Graph Manager**
2. **New Report** → Style: **Cartesian Position & Velocity**
3. Set frame: **J2000**
4. Set time step: **60 seconds** (1 minute)
5. Set time bounds to match your scenario interval
6. Generate → **Export to File** as CSV
7. Save as: `ISS_Model_J2000_Position_Velocity.csv`

**Expected columns:** `Time (UTCG), x (km), y (km), z (km), vx (km/sec), vy (km/sec), vz (km/sec)`

### 7.3 Report 2 — Sun Vector (J2000)

**Target object:** ISS satellite

1. **Report & Graph Manager → New Report** → Style: **Sun Vector**
2. Frame: **J2000**; Reference: **Sun** (as a central body position)
3. Time step: **60 seconds**
4. Generate → Export as CSV
5. Save as: `ISS_Model_Sun_Vector_J2000.csv`

**Expected columns:** `Time (UTCG), x (km), y (km), z (km)`

### 7.4 Report 3 — LLA Position

**Target object:** ISS satellite

1. **Report & Graph Manager → New Report** → Style: **LLA Position**
2. Time step: **60 seconds**
3. Generate → Export as CSV
4. Save as: `ISS_Model_LLA_Position.csv`

**Expected columns:** `Time (UTCG), Lat (deg), Lon (deg), Alt (km), Lat Rate (deg/sec), Lon Rate (deg/sec), Alt Rate (km/sec)`

### 7.5 Report 4 — Lighting Times

**Target object:** ISS satellite

1. **Report & Graph Manager → New Report** → Style: **Lighting Intervals** (or **Lighting Times**)
2. This generates three tables: Sunlight, Penumbra, Umbra
3. Generate → Export as a **single CSV file** (all three tables must appear sequentially in one file)
4. Save as: `ISS_Model_Lighting_Times.csv`

**Expected structure:**
```
[Sunlight table header]
Start Time (UTCG), Stop Time (UTCG), Duration (sec)
...rows...
[Global Statistics block]
[Penumbra table header]
Start Time (UTCG), Stop Time (UTCG), Duration (sec)
...rows...
[Global Statistics block]
[Umbra table header]
...rows...
```

> **Important:** The parser in `streamlit_app.py` and the notebook expects these three tables in this exact order (sunlight → penumbra → umbra). If your STK version orders them differently, adjust the `parse_lighting_report()` function accordingly.

### 7.6 Report 5 — Az/El Mask (Structural Blocking)

**Target object:** MSC sensor (or the relevant field-of-view sensor on the ISS)

> **Note:** This report is generated **once** from the physical STK sensor geometry and does **not** change when you change the simulation time period. It only changes if the sensor definition or ISS structural model changes. You can reuse `MSC_Sun_Sensor_Az_El_Mask.csv` across all simulation runs unless the sensor geometry changes.

If you do need to regenerate it:
1. In the STK MSC sensor object, open the **Field of View** properties
2. Use the **Az/El Mask Report** generator
3. Export the exclusion zone and inclusion zone polygon vertices
4. Save as: `MSC_Sun_Sensor_Az_El_Mask.csv`

**Expected structure:** Named zone blocks with `Point, Azimuth (deg), Elevation (deg)` rows (see §4.5 above for format details).

### 7.7 Placing New Reports in the Repository

After generating all reports, place them in:
```
simulation/msc-esh-6-month-sim/data/
```

If you want to keep old and new reports side by side, create a subfolder:
```
simulation/msc-esh-6-month-sim/data/2027-sim/
```
Then point the Streamlit sidebar **Data directory** field to the new subfolder, or update `DATA_DIR` at the top of the notebook.

### 7.8 Clearing the Albedo Cache

NASA POWER albedo values are cached per (lat, lon, date). When you change the simulation period, new dates will be fetched automatically. However, to force a complete re-fetch (e.g., to switch from ALLSKY to CLRSKY), delete the cache:
```bash
rm simulation/msc-esh-6-month-sim/data/cache/power_*.json
```

---

## 8. Key Configurable Parameters

### Physical constants (rarely changed)

| Parameter | Location | Value | Description |
|---|---|---|---|
| `AM0` | `streamlit_app.py`, notebook | `1361.0 W/m²` | Solar constant at 1 AU |
| `R_EARTH_KM` | `streamlit_app.py`, notebook | `6378.137 km` | WGS84 mean Earth radius |

### Albedo configuration

| Parameter | Default | Description |
|---|---|---|
| `ALBEDO_MODE` | `"power"` | `"power"` or `"constant"` |
| `ALBEDO_CONSTANT` | `0.27` | Fixed albedo; used in constant mode or as POWER fallback |
| `POWER_PARAM` | `"ALLSKY_SRF_ALB"` | `ALLSKY_SRF_ALB` = cloud-inclusive; `CLRSKY_SRF_ALB` = cloud-free |
| `query_interval_minutes` | `1440` | Minimum time between POWER API queries (1440 = once per day) |

### Exposure windows

Operational open intervals, entered as a list of `(start, end)` string tuples:
```python
EXPOSURE_WINDOWS = [
    ("01 Jul 2020 12:00:00.000", "01 Jul 2020 14:00:00.000"),
    ("02 Jul 2020 10:00:00.000", "02 Jul 2020 12:00:00.000"),
]
```
Leave as `[]` for always-open (no exposure masking).

### Timestep (STK export cadence)

The simulation timestep is determined by the STK export cadence, **not** a parameter in the Python code. To change from 1-minute to 30-second timesteps:
- Regenerate all STK reports at 30-second intervals
- The simulation code reads `dt` dynamically from the timestamp differences — no code change required

---

## 9. Albedo Options

Earth surface albedo is the fraction of incoming solar radiation reflected back from Earth's surface toward the spacecraft. It drives the **albedo irradiance** component:

```
I_albedo = ρ_eff × AM0 × F_earth × day_factor × cos(θ_earth_to_face)
```

where `F_earth = sin²(arcsin(R_earth / r_spacecraft))` is the Earth disc view factor.

### Option A: NASA POWER API (recommended)

- **Parameter:** `ALLSKY_SRF_ALB` (all-sky, includes cloud reflection) — recommended for most cases
- **Parameter:** `CLRSKY_SRF_ALB` (clear-sky only) — use for conservative cloud-free estimate
- **Resolution:** 0.5° × 0.5° spatial, daily temporal
- **Caching:** Results cached to `data/cache/` as JSON; subsequent runs are instant
- **Fallback:** If POWER returns a fill value for a given cell, `ALBEDO_CONSTANT` (0.27) is used

### Option B: Constant Albedo

- Uses a single fixed value for all locations and times
- Fast, no internet required
- Global average Earth albedo ≈ 0.30; ISS orbit includes high-reflectivity ocean/cloud regions; 0.27 is a reasonable conservative estimate

---

## 10. Understanding the Three Masking Layers

The simulation applies three independent binary/scalar masks to the irradiance at each timestep. All three must be non-zero for any light to contribute to ESH:

```
I_direct_masked  = I_direct  × eclipse_factor × structural_factor × exposure_open_factor
I_albedo_masked  = I_albedo  × eclipse_factor ×        1.0        × exposure_open_factor
I_total_masked   = I_direct_masked + I_albedo_masked
```

Note that **structural blocking applies only to direct (collimated) sunlight**, not to diffuse albedo irradiance, which arrives from the whole Earth disc.

### Mask 1: Eclipse Factor

| Lighting state | Factor |
|---|---|
| Sunlight | 1.0 |
| Penumbra | 0.5 |
| Umbra | 0.0 |

Parsed from `ISS_Model_Lighting_Times.csv`. Penumbra events last ~12 seconds each and contribute negligibly to total ESH.

### Mask 2: Structural Factor

| Sun position | Factor |
|---|---|
| Blocked by ISS structure | 0.0 |
| Unblocked | 1.0 |

Computed by checking whether the Sun's (Az, El) in the sensor frame falls inside an exclusion zone polygon but outside all inclusion zone polygons within it. Uses `matplotlib.path.Path.contains_points()`.

**Beta angle sensitivity:** All structural exclusion zones are bounded at elevation ≤ 35.84°. When the orbital beta angle β > ~36°, the Sun remains above all exclusion zones for the entire orbit → structural_factor = 1.0 for all timesteps. This is **physically correct**, not a bug.

### Mask 3: Exposure Open Factor

| Operational state | Factor |
|---|---|
| Within an exposure window | 1.0 |
| Outside all exposure windows | 0.0 |

If `EXPOSURE_WINDOWS = []`, all timesteps get factor 1.0 (always open).

---

## 11. Output Files and Downstream Use

### Main output: `msc_esh_6month_irradiance.csv`

| Column | Units | Description |
|---|---|---|
| `t` | ISO 8601 string | Timestamp |
| `I_global` | W/m² | Total masked irradiance (direct + albedo, all 3 masks applied) |
| `export_face` | string | Face label (e.g., `+H`) |
| `sun_factor` | — | Eclipse factor (alias; 1.0 / 0.5 / 0.0) |
| `I_global_sunmasked` | W/m² | Backward-compat alias for `I_global` |
| `lighting_state` | string | `"Sunlight"`, `"Penumbra"`, or `"Umbra"` |
| `eclipse_factor` | — | 1.0 / 0.5 / 0.0 |
| `sun_az_deg` | degrees | Sun azimuth in sensor frame |
| `sun_el_deg` | degrees | Sun elevation in sensor frame |
| `structural_blocked` | bool | True if blocked by ISS structure |
| `structural_factor` | — | 0.0 (blocked) or 1.0 (unblocked) |
| `exposure_open_factor` | — | 0.0 or 1.0 |

### ESH calculation from the output CSV

```python
import pandas as pd
df = pd.read_csv("outputs/msc_esh_6month_irradiance.csv", parse_dates=["t"])
df = df.sort_values("t")
dt_sec = df["t"].diff().dt.total_seconds().fillna(60.0)
AM0 = 1361.0
df["ESH_increment"] = df["I_global"] * dt_sec / AM0 / 3600
total_ESH = df["ESH_increment"].sum()
```

### Downstream consumers

The primary downstream consumer (`sensor_channel_split.ipynb`) reads the five core columns:
- `t`, `I_global`, `export_face`, `sun_factor`, `I_global_sunmasked`

All other columns are diagnostic and may be ignored downstream.

---

## 12. Scaling the System

### Running multiple faces simultaneously

The simulation engine already computes all 6 faces in a single run. The `esh_summary` result dict (and the CSV) contain ESH for all faces. The `PRIMARY_FACE` / sidebar selection controls only which face is highlighted in plots — all faces are always calculated.

### Extending to a longer or different simulation period

1. Change the STK scenario analysis interval
2. Regenerate the four time-dependent reports (positions, sun vector, LLA, lighting times) — see §7
3. The Az/El mask is static and does not need regeneration
4. Point the simulation at the new CSV files
5. Run — no code changes required

### Running multiple scenarios in batch (programmatic)

```python
import sys
sys.path.insert(0, "simulation/msc-esh-6-month-sim")
from streamlit_app import run_simulation

scenarios = [
    {"data_dir": "data/2026-jan-jun", "primary_face": "+H"},
    {"data_dir": "data/2026-jul-dec", "primary_face": "+H"},
]

base_config = {
    "albedo_mode": "power",
    "albedo_constant": 0.27,
    "power_param": "ALLSKY_SRF_ALB",
    "power_cache_dir": "data/cache",
    "exposure_windows": [],
}

results = []
for s in scenarios:
    config = {**base_config, **s}
    result = run_simulation(config, log=print)
    results.append(result)
```

### Changing the STK time step

Export STK reports at any cadence you choose (e.g., 30 s, 2 min). The simulation reads `dt` dynamically from consecutive timestamp differences — no Python code changes needed. Finer cadence improves eclipse transition accuracy but increases memory and runtime.

### Adding a new face or changing face geometry

Face normals are defined in `streamlit_app.py` in the `FACE_NORMALS` dictionary (and equivalently in the notebook). To add or update a face:

```python
FACE_NORMALS = {
    "+R": np.array([ 1,  0,  0]),
    "-R": np.array([-1,  0,  0]),
    "+T": np.array([ 0,  1,  0]),
    "-T": np.array([ 0, -1,  0]),
    "+H": np.array([ 0,  0,  1]),
    "-H": np.array([ 0,  0, -1]),
    # Add a canted face:
    "+R_canted": np.array([0.866, 0.5, 0.0]),  # 30° cant toward +T
}
```

> **Note:** Current face normals are a generic LVLH box model. Real MSC panel orientations (tilt, cant, sensor boresight alignment) should be substituted once verified against the STK sensor mount definition.

---

## 13. Known Limitations and Future Work

| Limitation | Status | Impact |
|---|---|---|
| Face normals are generic LVLH box | Known; not yet fixed | Irradiance per face is approximate; correct to within typical alignment uncertainties |
| Sensor frame alignment unverified | To-do | Could result in Az/El offset vs. actual sensor boresight |
| Penumbra factor = 0.5 (placeholder) | Accepted; <0.01% ESH impact | Negligible |
| No ISS attitude variations | Known | Maneuvers (TEA mode, reboosts) not modeled |
| Albedo view-factor model is first-order | Known | F_earth and day_factor use simplified geometry |
| Structural mask polygons assume fixed body pointing | Known | Real solar array rotation not modeled |

---

## 14. Environment Setup

### Initial setup (first time)

```bash
cd /Users/tavishka/esh-estimate-simulation-aegis
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy matplotlib scipy requests jupyter
```

### Every subsequent session

```bash
source .venv/bin/activate
```

### Key Python dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pandas` | Data loading, time series, DataFrames |
| `numpy` | Vectorised math, LVLH frame computation |
| `matplotlib` | Plotting; `matplotlib.path.Path` for Az/El polygon containment |
| `requests` | NASA POWER API calls |
| `scipy` | (Optional) statistical utilities |
| `jupyter` / `jupyterlab` | Notebook execution |
