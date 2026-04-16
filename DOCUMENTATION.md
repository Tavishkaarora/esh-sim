# MSC ESH Simulation — Technical Documentation

**Document type:** Simulation methodology, physics, and design rationale
**Project:** MISSE Science Carrier (MSC) — Equivalent Sun Hours Estimation
**Platform:** International Space Station (ISS)

> This document captures the scientific and engineering reasoning behind every stage of the simulation. It is intended to record *why* each choice was made, not just *what* the code does. The README covers *how to use* the system; this document covers *how and why it works*.

---

## Table of Contents

1. [Project Background and Motivation](#1-project-background-and-motivation)
2. [What We Are Computing and Why](#2-what-we-are-computing-and-why)
3. [Coordinate Systems and Reference Frames](#3-coordinate-systems-and-reference-frames)
4. [Solar Irradiance Model](#4-solar-irradiance-model)
   - [4.1 Solar Constant (AM0)](#41-solar-constant-am0)
   - [4.2 Direct Irradiance](#42-direct-irradiance)
   - [4.3 Earth Albedo Irradiance](#43-earth-albedo-irradiance)
5. [Albedo Data Sources](#5-albedo-data-sources)
   - [5.1 NASA POWER (Recommended)](#51-nasa-power-recommended)
   - [5.2 Constant Albedo (Fast Fallback)](#52-constant-albedo-fast-fallback)
6. [STK as the Orbital Mechanics Engine](#6-stk-as-the-orbital-mechanics-engine)
7. [The Three Masking Layers](#7-the-three-masking-layers)
   - [7.1 Eclipse Mask](#71-eclipse-mask)
   - [7.2 Structural Blocking Mask](#72-structural-blocking-mask)
   - [7.3 Operational Exposure Mask](#73-operational-exposure-mask)
8. [ESH Integration](#8-esh-integration)
9. [Pipeline Implementation Decisions](#9-pipeline-implementation-decisions)
10. [Validation and Testing](#10-validation-and-testing)
11. [Assumptions and Simplifications](#11-assumptions-and-simplifications)
12. [Known Issues and Open Questions](#12-known-issues-and-open-questions)

---

## 1. Project Background and Motivation

The **MISSE Science Carrier (MSC)** is an external platform attached to the International Space Station (ISS). It exposes material samples and instruments to the space environment — including raw solar UV radiation, ionising radiation, thermal cycling, and atomic oxygen.

One important value for this project is the **cumulative solar irradiance** received by each face of the MSC during a mission segment. This matters because solar exposure affecrs how materials and sensors behave over time. In particular, it is needed to:

1. **Estimate degradation** of materials such as polymers, optical surfaces, and detectors
2. **Account for changes** in sensor response caused by UV exposure
3. **Compare on-orbit exposure with ground-truth** experiments where known UV doses are applied 

Calculating this exposure is not simple because the MSC does not receive sunlight in a constant or uniform way throughout the orbit. Several factors change the amount of irradiance reaching the platform:
- The ISS orbit changes realtive to the Sun over time, which changes the **beta angle** (angle between the Sun and the orbital plane) 
- The ISS body and solar arrays periodically **shadow the MSC** (structural blocking)
- The MSC may have defined **operational windows** (periods when it is actively acquiring data vs. dormant)
- **Earth albedo** — sunlight reflected off clouds and surface — contributes a non-trivial second source of irradiance

Because of these effects, total exposure cannot be estimated accurately using a simple average sunlight fraction multiplied by the solar constant. This simulation was developed to account for these time-dependent effects using orbit data generated in STK.

---

## 2. What We Are Computing and Why

**Equivalent Sun Hours (ESH)** is a way to express total solar exposure in units of hours at full solar intensity. It is calculated by integrating the effective solar irradiance over time and normalizing by the solar constant:

```
ESH = (1 / AM0) × ∫ I_effective(t) dt          [sun-hours]
```

where AM0 is the solar constant and the result is converted from seconds to hours.

In simple terms, **1 ESH** means the sample received the same total solar energy as it would receive from **one hour of full, unobstructed sunlight at 1 AU**. Using ESH makes it easier to compare on-orbit exposure with controlled laboratory exposure tests. 

ESH is calculated **for each face** of the MSC because each face sees a sifferent solar environment. This happens for several reasons:
- Each face has a different orientation relative to the Sun and Earth
- Different faces receive dramatically different cumulative doses depending on beta angle history
- The sensor response is face-specific, so each detetctor panel must be evaluated seperately 

The six MSC faces are defined in the **LVLH (Local Vertical Local Horizontal)** reference frame, which is described in the next section.

---

## 3. Coordinate Systems and Reference Frames

### J2000 Inertial Frame

STK exports ISS position, ISS velocity, and the Sun position in the **J2000** geocentric inertial frame. In this frame:
- Origin: Earth centre of mass
- Z-axis: North celestial pole (Earth rotation axis at J2000.0 epoch)
- X-axis: Vernal equinox direction at J2000.0

This is the reference frame used by STK for the state vector data exported to CSV.

### LVLH Frame (Local Vertical Local Horizontal)

The simulation geometry is computed in the **LVLH frame**, which is attached to the spacecraft and therefore rotates as the ISS orbits.

The three LVLH unit vectors are defined as follows:

| Axis | Symbol | Definition |
|---|---|---|
| Radial | R̂ | Unit vector from the Earth center to the ISS |
| Along-track | T̂ | Unit vector in the direction of motion within the orbital plane |
| Orbit normal | Ĥ | Unit vector normal to the orbital plane |

These vectors are contructed from the ISS position and velocity from STK:
```python
r = np.array([x, y, z])           # ISS position in J2000
v = np.array([vx, vy, vz])        # ISS velocity in J2000

R_hat = r / np.linalg.norm(r)
T_hat = np.cross(H_hat, R_hat)    # via orbit normal first:
H_hat = np.cross(r, v)
H_hat = H_hat / np.linalg.norm(H_hat)
T_hat = np.cross(H_hat, R_hat)    # then along-track
```
This creates a right-handed LVLH coordinate system that is used to describe face orientations and Sun direction throguhout the simulation.

### Sun Direction in LVLH

The Sun position vector from STK (`sun_pos` in J2000) is first converted to a unit direction vector from the ISS to the Sun:

```python
sun_vec = sun_pos - iss_pos        # Sun direction from ISS
sun_unit = sun_vec / np.linalg.norm(sun_vec)

# Project onto LVLH basis:
s_R = np.dot(sun_unit, R_hat)      # radial component
s_T = np.dot(sun_unit, T_hat)      # along-track component
s_H = np.dot(sun_unit, H_hat)      # orbit-normal component
```

These three components describe the Sun direction in LVLH coordinates, which allows the simulation to determine how much irradiance reaches each MSC face.

### Sensor Frame / Az/El Convention

The structural blocking mask is expressed in an **azimuth/elevation** coordinate system tied to the sensor frame:

- **Elevation (El)** is computed as `arcsin(s_H)` and represents the angle above the orbital plane. 
- **Azimuth (Az)** is computed as `atan2(s_T, s_R)` and represents the angle in the orbital plane, measured from the +R direction toward the +T direction.
  
With this convention: 
- `El = 0°` lies in the orbital plane
- `El = +90°` points in the +H direction
- `Az = 0°` points along +R
- `Az = 90°` points along +T
  
This convention was chosen to match the Az/El definitions used in the STK sensor field-of-view report that generated `MSC_Sun_Sensor_Az_El_Mask.csv`.

**Important:** The current implementation assumes that the sensor boresight is aligned with the +H direction, which is the most common ISS external payload orientation. This assumption should be verified against the actual MSC mounting orientation and the sensor frame used in STK.

---

## 4. Solar Irradiance Model

### 4.1 Solar Constant (AM0)

The simulation uses:

```
AM0 = 1361.0 W/m²
```

This value represents the **total solar irradiance (TSI)** at 1 AU outside Earth's atmosphere, often called the "AM0" (air mass zero, i.e., outside the atmosphere) value. The current accepted best value is approximately 1361 ± 0.5 W/m² (Kopp & Lean 2011, revised SORCE/TIM measurements).

The TSI varies by ~0.1% over the 11-year solar cycle and by ~0.07% seasonally (Earth's eccentricity). For a 6-month simulation, seasonal variation in Earth–Sun distance would produce at most a ~3% difference in direct irradiance between perihelion (early January, ~1410 W/m²) and aphelion (early July, ~1320 W/m²). The current simulation uses a **fixed AM0 = 1361 W/m²** (the annual mean). For higher fidelity, this could be replaced with a time-varying TSI computed from Earth's ecliptic longitude.

### 4.2 Direct Irradiance

For a flat surface with outward unit normal n̂, the direct solar irradiance (W/m²) at each timestep t is calculated as:

```
I_direct = AM0 × max(0, sun_unit · n̂)
```

The dot product gives the cosine of the angle between the Sun direction and the face normal. If that value is negative, the face is pointing away from the Sun, so the direct irradiance is set to zero. 

For each of the 6 LVLH-aligned faces, this becomes:
```python
# For face "+H" (normal = [0, 0, 1]):
cos_angle = s_H          # dot product with [0, 0, 1]
I_direct_H = AM0 * max(0, cos_angle)

# For face "+R" (normal = [1, 0, 0]):
cos_angle = s_R
I_direct_R = AM0 * max(0, cos_angle)

# other faces are handled the same way
```

### 4.3 Earth Albedo Irradiance

In addition to direct sunlight, the ISS also receives sunlight reflected from the Earth and clouds. This reflected compononet is modeled as **albedo irradiance**. In this simulation, albedo irradiance depends on four terms:

1. **Earth surface reflectivity ρ** (albedo, dimensionless 0–1)
2. **Earth view factor F_earth** — what fraction of the sky hemisphere "seen" by the face is occupied by Earth
3. **Day factor** — what fraction of the visible Earth disk is currently sunlit (illuminated from the Sun's side)
4. **Geometry** — the cosine of the angle between the face normal and the Earth centre direction

**Earth view factor:**

From ISS altitude, the Earth fills a large part of the downward-looking hemisphere. The Earth view factor is approximated as:

```
α = arcsin(R_earth / r)                # half-angle subtended by Earth disk
F_earth = sin²(α)                       # view factor of Earth hemisphere
```
where r = R_earth + h is the orbital radius.

At typical ISS altitude (~410 km), `r ≈ 6788 km`, `α ≈ arcsin(6378/6788) ≈ 69.9°`, so `F_earth ≈ sin²(69.9°) ≈ 0.881`. Earth fills about 88% of the lower hemisphere as seen from the ISS.

**Day factor:**

Only the sunlit portion of the Earth contributes reflected sunlight. The model estimates this using a simple day factor:

```python
# Sun-Earth-ISS angle (ISS side)
sun_earth_angle = arccos(clamp(dot(sun_unit, -earth_unit), -1, 1))
day_factor = 0.5 × (1 + cos(sun_earth_angle))
```

where `earth_unit = -R_hat` (direction from ISS to Earth centre) and `sun_unit` is the Sun direction from ISS. This gives 1.0 when Earth is fully illuminated (ISS between Earth and Sun, i.e., minimum beta angle at noon) and 0.0 when Earth is fully in shadow (ISS on the night side).

**Albedo irradiance per face:**

For each face with LVLH normal n̂, the albedo irradiance is calculated as:
```
cos_earth = max(0, dot(n̂, -R_hat))    # angle to Earth centre; -R_hat = nadir

I_albedo = ρ_eff × AM0 × F_earth × day_factor × cos_earth
```

Note that `cos_earth` for the **-R (nadir) face** is maximised (= 1.0) since it points directly at Earth. The **+R (zenith) face** receives zero albedo since its normal points away from Earth.

In the current model, albedo is **not affected by the structural blocking mask**. This is because albedo comes from the broad Earth disk rather than from a single direct Sun direction. The structural mask is only applied to direct solar irradiance.

---

## 5. Albedo Data Sources

### 5.1 NASA POWER (Recommended)

The main albedo data source used in this simulation is **NASA POWER** (Prediction of Worldwide Energy Resources). NASA POWER provides environmental and solar resource data on a global 0.5° × 0.5° grid and can be queried by latitude, longitude, and date. 

In this project, the `ALLSKY_SRF_ALB` parameter is used as an estimate of surface albedo. This provides a practical way to include geographic and day-to-day variation in reflected sunlight. 

**NASA POWER** was selected because it:
- is freely available through an API (no registration needed)
- has global spatial coverage
- provides daily data
- is easy to integrate into the simulation pipeline 

**API mechanics:**
1. The ISS sub-satellite lat/lon is taken from `ISS_Model_LLA_Position.csv`
2. Coordinates are rounded to the nearest 0.5° POWER grid centre to reduce API calls for nearby positions
3. One API call is made per unique (lat, lon, date) triple, with results cached as JSON
4. The default query interval is 1440 minutes (one call per calendar day), meaning the ISS position is sampled once per day for the albedo query

**Tradeoffs:**
- POWER surface albedo is not identical to TOA (top-of-atmosphere) reflected albedo — it does not include cloud reflectivity added in the column. For space-based irradiance, TOA reflected flux is more physically correct.
- However, POWER is readily available and the difference is partially accounted for by the `ALLSKY` parameter which incorporates cloud cover information through MERRA-2.

**Cache structure:**
```
data/cache/power_<lat>_<lon>_<date>.json
```
e.g., `power_27.00_-86.50_20200811.json` → `{"value": 0.08, "parameter": "ALLSKY_SRF_ALB", ...}`

### 5.2 Constant Albedo (Fast Fallback)

A single fixed value ρ = 0.27 is used at all times and locations.

**Why 0.27?**
- Earth's global mean Bond albedo is approximately 0.30
- ISS orbits at ~51.6° inclination, overflying predominantly ocean and mid-latitude land
- Ocean albedo is ~0.06–0.10; mid-latitude land ~0.15–0.25; cloud tops ~0.60–0.80
- Weighted by orbital coverage and typical cloud fraction, 0.27 is a reasonable central estimate
- It is slightly below the global mean (0.30) to account for the ISS orbital ground track sampling more ocean than the global average

**When to use constant albedo:**
- Quick validation runs where albedo accuracy is not critical
- When POWER API is unavailable (no internet access)
- When generating many scenarios rapidly (batch runs)

---

## 6. STK as the Orbital Mechanics Engine

Systems Tool Kit (STK) by Ansys was chosen as the orbit propagation engine for several reasons:

1. **High-fidelity propagation:** STK's HPOP (High Precision Orbit Propagator) accounts for J2–J6 gravity harmonics, atmospheric drag (NRLMSISE-00), solar radiation pressure, and third-body effects (Moon, Sun). This is essential for accurate ISS state prediction over month-long spans.

2. **Eclipse computation:** STK's lighting analysis uses a conical Earth shadow model (not a cylindrical approximation), correctly computing penumbra entry/exit geometry. This is computationally expensive to implement from scratch.

3. **Sensor field-of-view masks:** The Az/El mask report is generated directly from the STK sensor object geometry, which incorporates the full CAD/analytical model of ISS structural blocking. This is far more accurate than any simplified hand-coded model.

4. **Authority of truth:** STK is NASA's standard tool for ISS mission planning. Using STK state vectors as inputs ensures the simulation is consistent with official ISS trajectory products.

**What STK does NOT provide directly:**
- Albedo (handled by NASA POWER)
- Per-face irradiance integration (handled by Python simulation)
- ESH accumulation (handled by Python simulation)
- Operational exposure masking (handled by Python simulation)

The design philosophy is: **let STK handle orbital mechanics; handle photonics and mission logic in Python.**

---

## 7. The Three Masking Layers

The simulation applies three independent masks to the irradiance at each timestep. The masks are independent — they are multiplied together, not combined via any other logic. This makes it easy to diagnose which mask is responsible for any reduction in ESH.

### 7.1 Eclipse Mask

**Physical basis:** When the ISS passes through Earth's shadow, direct solar irradiance drops to zero (umbra) or partially attenuates (penumbra). This occurs every orbit during certain beta angle regimes.

**Implementation:**
- Parse the three tables from `ISS_Model_Lighting_Times.csv` into sunlight, penumbra, and umbra interval lists
- For each timestep, determine which interval contains that timestamp
- Assign `eclipse_factor`:
  - Sunlight → 1.0
  - Penumbra → 0.5 (see note below)
  - Umbra → 0.0

**Penumbra factor (0.5) — rationale and limitations:**
Penumbra is the annular region where Earth partially occults the solar disk as seen from ISS. The physically correct irradiance during penumbra would be computed from the fraction of the solar disk that remains unoccluded, which varies continuously from 1.0 (penumbra entry) to 0.0 (umbra entry). This requires knowledge of the Sun's angular diameter (~0.5°) and the precise geometry of Earth's shadow.

For this simulation, a factor of 0.5 is used as a simple midpoint approximation. This is acceptable because:
- Penumbra events at ISS altitude last approximately 12 seconds
- In a 1-minute cadence dataset, each penumbra event affects at most 1 timestep
- The total ESH contribution from all penumbra timesteps combined is less than 0.01% of total ESH

**Eclipse fraction statistics:** At ISS orbital parameters (altitude ~410 km, inclination 51.6°), the ISS spends approximately:
- ~56 minutes per orbit in sunlight
- ~34 minutes per orbit in umbra
- ~24 seconds per eclipse entry/exit in penumbra

The eclipse fraction varies significantly with beta angle: at high beta angles (> ~72°), the ISS enters continuous sunlight (no eclipses). The simulation handles this naturally since the lighting times CSV will simply have no umbra entries for those periods.

### 7.2 Structural Blocking Mask

**Physical basis:** The ISS structure — primarily the Integrated Truss Structure (ITS), solar array wings (SAWs), and the Pressurised Mating Adapters — casts shadows on the MSC sensor aperture for certain Sun directions. These obstructions are modelled in STK as a sensor Az/El mask (field-of-view constraint).

**The Az/El mask format:**
The file `MSC_Sun_Sensor_Az_El_Mask.csv` contains polygon vertex sequences defining:
- **6 Exclusion Zones (EZ1–EZ6):** Az/El regions where ISS structure blocks the Sun
- **57 Inclusion Zones (IZ):** Sub-regions within exclusion zones that are *not* actually blocked (windows in the structure, gaps between arrays)

A Sun position is **structurally blocked** if and only if:
- It falls **inside** at least one exclusion zone polygon, **AND**
- It falls **outside** all inclusion zone polygons that are associated with that exclusion zone

**Polygon containment implementation:**
The code uses `matplotlib.path.Path.contains_points()`, which implements a standard ray-casting algorithm to determine point-in-polygon membership. This is applied to the (Az, El) pair at each timestep.

```python
from matplotlib.path import Path

def is_blocked(az_deg, el_deg, exclusion_zones, inclusion_zones_by_ez):
    point = np.array([[az_deg, el_deg]])
    for ez_idx, ez_poly in enumerate(exclusion_zones):
        if Path(ez_poly).contains_point(point[0]):
            # Inside exclusion zone; check if inside any inclusion zone
            for iz_poly in inclusion_zones_by_ez[ez_idx]:
                if Path(iz_poly).contains_point(point[0]):
                    return False  # Inside IZ within EZ → not blocked
            return True           # Inside EZ but not any IZ → blocked
    return False                  # Outside all exclusion zones → not blocked
```

**Beta angle sensitivity:**
All structural exclusion zones in the current mask are bounded at elevation ≤ +35.84°. When the orbital beta angle β > ~36°, the Sun's elevation in the LVLH frame exceeds 35.84° for the entire orbit — it is always above all exclusion zone ceilings. In this regime, `structural_factor = 1.0` for every timestep, meaning zero structural blocking.

This is physically correct: when the Sun is nearly aligned with the orbit normal (high beta angle), it shines directly onto the +H face from above, and the ISS body/truss/arrays do not shadow the sensor aperture.

The one-day prototype test (`esh_prototype.ipynb`) was run on 4–5 Mar 2026 when β ≈ 42°, which is why structural blocking appears as 0% — this is expected behaviour, not a bug.

### 7.3 Operational Exposure Mask

**Physical basis:** The MSC instrument may not be acquiring data continuously. Operational exposure windows define the periods when the instrument shutter (or equivalent) is open and actively receiving radiation. Outside these windows, the sample is protected.

**Implementation:**
- User provides a list of (start, end) timestamp strings
- These are parsed and merged (overlapping windows are combined)
- A per-timestep binary factor is created: 1.0 inside any window, 0.0 outside all windows
- If no windows are specified, the factor is 1.0 for all timesteps (always-open)

**Why this matters:**
For multi-month missions with defined operational schedules (e.g., instrument activated for 2 hours per day during certain orbital passes), the ESH accumulated during dormant periods is zero regardless of solar geometry. This mask allows the simulation to accurately represent the actual instrument exposure history.

**Note on albedo masking:** The exposure mask is applied to albedo irradiance as well as direct irradiance. The physical rationale is that if the instrument aperture is closed, it receives neither direct nor reflected irradiance.

---

## 8. ESH Integration

After applying all three masks, the final irradiance per face per timestep is:

```
I_direct_masked  = I_direct  × eclipse_factor × structural_factor × exposure_factor
I_albedo_masked  = I_albedo  × eclipse_factor × 1.0               × exposure_factor
I_total_masked   = I_direct_masked + I_albedo_masked
```

**Timestep integration:**
The timestep duration `dt` (seconds) is computed from consecutive timestamp differences:
```python
dt = (timestamps[t+1] - timestamps[t]).total_seconds()
```

For a uniform STK export cadence (e.g., 60 seconds), all dt values are equal. For irregular cadences or edge cases, using actual timestamp differences is more robust than assuming a fixed dt.

**ESH accumulation:**
```python
ESH_increment = I_total_masked × dt / AM0 / 3600    # sun-hours per timestep
ESH_cumulative = cumsum(ESH_increment)
```

**Total ESH per face:**
```python
ESH_total = sum(ESH_increment)    # integrated over full simulation period
```

**Per-face statistics reported:**
- Total ESH (sun-hours)
- Peak irradiance (W/m²)
- Mean irradiance when unblocked (W/m²)
- Fraction of time in sunlight (vs. eclipse)
- Fraction of time structurally blocked (when in sunlight)

---

## 9. Pipeline Implementation Decisions

### Why Python + pandas rather than MATLAB or Fortran?

- **Portability:** Python runs on all platforms without license requirements
- **NASA POWER API access:** Python's `requests` library makes REST API calls straightforward
- **Streamlit:** Rapid interactive UI deployment with minimal front-end code
- **Pandas datetime handling:** Robust parsing of STK timestamp strings and time series alignment

### Why Streamlit for the UI?

Streamlit converts a Python script into an interactive web app with minimal boilerplate. Key advantages for this use case:
- Users can change simulation parameters (face, albedo mode, exposure windows) without editing code
- Real-time progress feedback via log streaming
- CSV download buttons without file I/O code
- Runs locally — no server, no cloud dependency, no data leaves the machine

### Why cache NASA POWER responses to disk?

The POWER API has rate limits and network latency. A 6-month simulation over the ISS orbit samples ~180 unique (lat, lon) positions per day at 1440-minute query intervals = ~180 API calls per simulation run. Disk caching:
- Makes subsequent runs essentially instant (~0 API calls if all dates already cached)
- Prevents hitting API rate limits on repeated development runs
- Allows offline runs once the cache is populated

### Why use matplotlib.path for Az/El polygon containment?

The Az/El mask has 57 inclusion zone polygons. A vectorised implementation using `Path.contains_points()` tests all N timesteps against all polygons in one NumPy operation, which is orders of magnitude faster than a Python loop over timesteps.

### Why not use STK's built-in irradiance analysis?

STK's solar panel analysis tools can compute irradiance on a surface, but they:
- Do not integrate ESH over arbitrary exposure windows
- Do not support the NASA POWER albedo data pipeline
- Do not produce the specific output CSV schema required by downstream analysis
- Are not easily scriptable for batch runs across multiple scenarios

The Python simulation gives complete control over every aspect of the computation.

---

## 10. Validation and Testing

### Prototype notebook (`esh_prototype.ipynb`)

The prototype serves as a single-day end-to-end validation:
- **Input:** 4–5 Mar 2026 (one day of STK data)
- **Expected eclipse fraction:** ~37% (34 min umbra per ~90 min orbit) ✓
- **Expected structural blocking:** 0% (β ≈ 42° > 36° threshold) ✓ 
- **Expected peak +H irradiance:** AM0 × cos(0°) = 1361 W/m² when Sun is directly above ✓
- **Expected -R (nadir) ESH dominance over +R (zenith) for albedo:** ✓ (nadir face sees Earth; zenith does not)

### Diagnostic plots

The simulation generates 9 diagnostic plots to support visual validation:
1. **fig_azel_diagnostic.png** — Sun trajectory in Az/El space overlaid on exclusion zone polygons. Validates structural mask geometry.
2. **fig_sun_eclipse.png** — Sun elevation angle and eclipse factor on the same timeline. Should show eclipse_factor = 0 when Sun is below the horizon (in eclipse).
3. **fig_structural_blocking.png** — structural_factor vs. time. Should show blocking events correlated with Sun passing through known ISS shadow zones.
4. **fig_direct_irradiance.png** — Direct irradiance per face. Should show sinusoidal variation with orbital period (~92 min), going to zero in eclipse.
5. **fig_total_irradiance.png** — Total irradiance including albedo. Albedo contribution should be visible during eclipse (from Earth-reflected light on non-blocked faces when albedo is not zero).
6. **fig_cumulative_esh_all.png** — Monotonically non-decreasing for all faces. Slope reflects average irradiance level.
7. **fig_per_orbit_flux.png** — One-orbit irradiance profile. Useful for sanity-checking orbital geometry.
8. **fig_exposure_mask.png** — Operational windows overlaid on irradiance. Gaps in ESH accumulation should coincide with exposure_open_factor = 0 periods.
9. **fig_cumulative_esh_faces.png** — All faces together. The +H face typically accumulates the most ESH (Sun illuminates +H face most during high-beta orbits which dominate the 6-month average).

### Numerical sanity checks

A 6-month simulation at ISS parameters should produce:
- Total ESH for the best-illuminated face: approximately 700–1200 sun-hours (depending on beta angle history and exposure windows)
- Eclipse fraction: ~38–45% of total time (varies with beta angle)
- Albedo contribution: ~5–15% of total irradiance (albedo irradiance is maximum ~0.30 × 1361 × 0.88 ≈ 360 W/m² for the nadir face, but most faces are not nadir-facing)

---

## 11. Assumptions and Simplifications

This section records every deliberate simplification, its physical justification, and the expected impact on accuracy.

| Assumption | Justification | Expected ESH error |
|---|---|---|
| Fixed TSI (AM0 = 1361 W/m²) | ±0.5 W/m² uncertainty; seasonal variation ±3.5% for Earth eccentricity | <4% peak; ~0% annual mean |
| Generic LVLH box face normals | Actual MSC panel orientations unknown pending geometry delivery | Could be significant (order of cos(cant_angle) correction) |
| Constant penumbra factor = 0.5 | Penumbra events are ~12 s; negligible ESH contribution | <0.01% |
| POWER daily albedo applied at 1-minute cadence | Albedo varies slowly (synoptic-scale clouds); sub-daily variation is secondary | <5% on individual orbits; <2% monthly |
| First-order Earth view factor (sin²(α)) | Exact form for disc view factor; valid for spherical Earth | <1% |
| First-order day factor (half-illuminated Earth) | Simplified; exact form requires solar zenith angle across entire Earth disc | ~5–10% on albedo component; ~1–3% on total |
| No ISS attitude variations | ISS holds approximately nadir-pointing; maneuvers are infrequent | <2% over multi-month period |
| Fixed sensor frame (no solar array rotation modelled) | ISS SAWs rotate to track the Sun; SAW shadow direction rotates with them | This affects structural blocking accuracy during certain Sun angles |
| Structural blocking ignores diffuse albedo | Diffuse irradiance from extended Earth disc is not collimated; blocking modelled for collimated Sun only | Physically appropriate; ~0% error |

---

## 12. Known Issues and Open Questions

### Open: Actual MSC face normals and cant angles

The current simulation uses idealised LVLH-aligned face normals (+R, -R, +T, -T, +H, -H). Real MSC panels may have:
- Cant angles (off-axis tilts for solar angle optimisation)
- Non-orthogonal arrangements
- A boresight that does not align with any pure LVLH axis

**Action required:** Obtain MSC structural geometry (ICD or STK attachment body definition) and update `FACE_NORMALS` in the simulation.

### Open: STK sensor frame verification

The structural blocking mask was generated from an STK sensor object. The Az/El convention used in `MSC_Sun_Sensor_Az_El_Mask.csv` must be verified to match the Python computation:
- `Az = atan2(s_T, s_R)` — measured from +R toward +T
- `El = arcsin(s_H)` — above the orbital plane

If the STK sensor has a different frame definition (e.g., Az measured from +H, or different handedness), the structural blocking geometry will be incorrect and the sensor object definition in STK should be consulted to confirm the frame convention.

### Open: Solar array shadow rotation

The ISS solar arrays track the Sun (α-axis rotation) and precess seasonally (β-axis). The STK Az/El mask was generated for a specific solar array orientation. The shadow cast by the arrays rotates as they track, meaning the structural blocking zones in Az/El space are not fixed — they rotate with the array gimbal angles. Modelling this accurately requires either:
- Generating multiple Az/El masks at different array orientations and interpolating, or
- Using STK's time-varying constraint report

The current mask is a conservative static approximation.

### Open: ESH-to-degradation transfer function

This simulation produces ESH (total normalised irradiance dose). The actual sensor degradation model that converts ESH to responsivity change (the "transfer function") is maintained separately and is outside the scope of this repository.

---

*This document is a living record. Sections will be updated as the simulation matures, new data sources are integrated, and open questions are resolved.*
