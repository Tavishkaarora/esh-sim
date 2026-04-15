"""
streamlit_app.py — MSC ESH Simulation UI

Interactive one-page app for the MSC Equivalent Sun Hours prototype.

Run with:
    cd simulation/msc-esh-6-month-sim
    streamlit run streamlit_app.py
"""

import re
import sys
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AM0          = 1361.0        # W/m²  AM0 solar irradiance
R_EARTH_KM   = 6378.137
STK_TIME_FMT = "%d %b %Y %H:%M:%S.%f"
VALID_FACES  = ["+R", "-R", "+T", "-T", "+H", "-H"]
FACE_NORMALS = {
    "+R": np.array([ 1., 0., 0.]),
    "-R": np.array([-1., 0., 0.]),
    "+T": np.array([ 0., 1., 0.]),
    "-T": np.array([ 0.,-1., 0.]),
    "+H": np.array([ 0., 0., 1.]),
    "-H": np.array([ 0., 0.,-1.]),
}
DEFAULT_DATA_DIR  = str(Path(__file__).parent / "data")
DEFAULT_CACHE_DIR = str(Path(__file__).parent / "data" / "cache")


# ---------------------------------------------------------------------------
# Pipeline helpers — geometry
# ---------------------------------------------------------------------------
def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Pipeline helpers — exposure windows
# ---------------------------------------------------------------------------
def parse_exposure_windows(text: str) -> list:
    """Parse 'start | stop' text (one window per line) into (Timestamp, Timestamp) list."""
    windows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 2:
            raise ValueError(
                f"Expected 'start | stop' format, got: {line!r}"
            )
        t_s = pd.to_datetime(parts[0], format=STK_TIME_FMT)
        t_e = pd.to_datetime(parts[1], format=STK_TIME_FMT)
        if t_e <= t_s:
            raise ValueError(f"End must be after start: {line!r}")
        windows.append((t_s, t_e))
    if not windows:
        return []
    windows.sort(key=lambda x: x[0])
    merged = [windows[0]]
    for s, e in windows[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def build_exposure_mask(t_series: pd.Series, windows: list) -> np.ndarray:
    if not windows:
        return np.ones(len(t_series), dtype=float)
    t_arr = t_series.to_numpy()
    mask  = np.zeros(len(t_series), dtype=bool)
    for ts, te in windows:
        mask |= (t_arr >= np.datetime64(ts)) & (t_arr <= np.datetime64(te))
    return mask.astype(float)


# ---------------------------------------------------------------------------
# Pipeline helpers — lighting report
# ---------------------------------------------------------------------------
def parse_lighting_report(path: str):
    """Return (sunlight_df, penumbra_df, umbra_df) as DataFrames with 'start'/'stop' columns."""
    HEADER = '"Start Time (UTCG)","Stop Time (UTCG)","Duration (sec)"'
    raw    = Path(path).read_text().splitlines()
    hdrs   = [i for i, l in enumerate(raw) if l.strip() == HEADER]

    def _extract(idx):
        si = hdrs[idx]
        ei = si + 1
        while ei < len(raw):
            l = raw[ei].strip()
            if l == "" or l.startswith("Global Statistics"):
                break
            ei += 1
        df = pd.read_csv(StringIO("\n".join(raw[si:ei])))
        df["start"] = pd.to_datetime(df["Start Time (UTCG)"], format=STK_TIME_FMT)
        df["stop"]  = pd.to_datetime(df["Stop Time (UTCG)"],  format=STK_TIME_FMT)
        return df[["start", "stop"]]

    return _extract(0), _extract(1), _extract(2)   # sunlight, penumbra, umbra


def build_eclipse_factor(t: np.ndarray, umbra_df, penumbra_df):
    in_umbra    = np.zeros(len(t), dtype=bool)
    in_penumbra = np.zeros(len(t), dtype=bool)
    for s, e in zip(umbra_df["start"].to_numpy(), umbra_df["stop"].to_numpy()):
        in_umbra    |= (t >= s) & (t <= e)
    for s, e in zip(penumbra_df["start"].to_numpy(), penumbra_df["stop"].to_numpy()):
        in_penumbra |= (t >= s) & (t <= e)
    ef     = np.ones(len(t))
    ef[in_penumbra] = 0.5
    ef[in_umbra]    = 0.0
    states = np.where(in_umbra, "Umbra", np.where(in_penumbra, "Penumbra", "Sunlight"))
    return ef, states


# ---------------------------------------------------------------------------
# Pipeline helpers — az/el mask
# ---------------------------------------------------------------------------
def parse_az_el_mask(path: str):
    with open(path) as f:
        lines = f.readlines()
    excl_zones = []
    incl_map   = {}
    cur_type   = None
    cur_excl   = None
    cur_pts    = []
    excl_re = re.compile(r'"Exclusion Zone (\d+)', re.IGNORECASE)
    incl_re = re.compile(r'"Inclusion Zone \d+ \(Inside Exclusion Zone (\d+)\)',
                         re.IGNORECASE)
    data_re = re.compile(r'\d+,(-?\d+\.?\d*),(-?\d+\.?\d*)')

    def _flush():
        if not cur_pts:
            return
        pts = np.array(cur_pts, dtype=float)
        if cur_type == "exclusion":
            excl_zones.append(pts)
        elif cur_type == "inclusion":
            incl_map.setdefault(cur_excl, []).append(pts)

    for line in lines:
        line = line.strip()
        me = excl_re.search(line)
        mi = incl_re.search(line)
        md = data_re.match(line)
        if me:
            _flush();  cur_type = "exclusion";  cur_excl = int(me.group(1)) - 1;  cur_pts = []
        elif mi:
            _flush();  cur_type = "inclusion";  cur_excl = int(mi.group(1)) - 1;  cur_pts = []
        elif md:
            cur_pts.append([float(md.group(1)), float(md.group(2))])
    _flush()
    return excl_zones, incl_map


def build_structural_factor(sun_az, sun_el, excl_zones, incl_map):
    excl_paths = [MplPath(z, closed=True) for z in excl_zones]
    incl_paths = {i: [MplPath(z, closed=True) for z in zs] for i, zs in incl_map.items()}
    pts        = np.column_stack([sun_az, sun_el])
    in_excl    = np.zeros(len(pts), dtype=bool)
    in_incl    = np.zeros(len(pts), dtype=bool)
    for i, ep in enumerate(excl_paths):
        in_e_i = ep.contains_points(pts)
        in_excl |= in_e_i
        for ip in incl_paths.get(i, []):
            in_incl |= (in_e_i & ip.contains_points(pts))
    blocked = in_excl & (~in_incl)
    return np.where(blocked, 0.0, 1.0), blocked


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------
def _build_export_df(df: pd.DataFrame, face: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "t":           df["t"],
        "I_global":    df[f"I_{face}"],
        "export_face": face,
        "sun_factor":  df["sun_factor"],
    })
    dc, ac = f"I_direct_{face}", f"I_albedo_{face}"
    if dc in df.columns and ac in df.columns:
        out["I_global_sunmasked"] = (
            df[dc] * df["eclipse_factor"] * df["structural_factor"] * df["exposure_open_factor"]
            + df[ac] * df["eclipse_factor"] * df["exposure_open_factor"]
        )
    else:
        out["I_global_sunmasked"] = df[f"I_{face}"]
    for col in ["lighting_state", "eclipse_factor", "sun_az_deg", "sun_el_deg",
                "structural_blocked", "structural_factor", "exposure_open_factor"]:
        if col in df.columns:
            out[col] = df[col]
    return out


# ---------------------------------------------------------------------------
# Core simulation pipeline
# ---------------------------------------------------------------------------
def run_simulation(config: dict, log) -> dict:
    """
    Run the full MSC ESH simulation and return a results dict.

    Parameters
    ----------
    config : dict
        data_dir, primary_face, exposure_windows, albedo_mode,
        albedo_constant, power_param, power_cache_dir, az_offset_deg
    log : callable(str)  — progress callback fed to the Streamlit status area.
    """
    dd = config["data_dir"]

    # ---- Load STK exports -----------------------------------------------
    log("Loading STK exports...")
    iss_df = pd.read_csv(f"{dd}/ISS_Model_J2000_Position_Velocity.csv")
    sun_df = pd.read_csv(f"{dd}/ISS_Model_Sun_Vector_J2000.csv")

    iss_df = iss_df.rename(columns={
        "Time (UTCG)": "time",
        "x (km)": "r_x", "y (km)": "r_y", "z (km)": "r_z",
        "vx (km/sec)": "v_x", "vy (km/sec)": "v_y", "vz (km/sec)": "v_z",
    })
    sun_df = sun_df.rename(columns={
        "Time (UTCG)": "time",
        "x (km)": "sun_x", "y (km)": "sun_y", "z (km)": "sun_z",
    })
    df = pd.merge(iss_df, sun_df, on="time")

    # ---- Timestamps + dt ------------------------------------------------
    df["t"]  = pd.to_datetime(df["time"], format=STK_TIME_FMT)
    df       = df.sort_values("t").reset_index(drop=True)
    df["dt"] = df["t"].diff().dt.total_seconds()
    df.loc[0, "dt"] = df.loc[1, "dt"]

    # ---- Exposure mask ---------------------------------------------------
    log("Building exposure mask...")
    df["exposure_open_factor"] = build_exposure_mask(
        df["t"], config.get("exposure_windows", [])
    )

    # ---- Albedo / rho_eff -----------------------------------------------
    mode     = config.get("albedo_mode", "constant")
    fallback = config.get("albedo_constant", 0.27)

    if mode == "constant":
        log(f"Albedo: constant ρ = {fallback:.3f}")
        df["rho_eff"] = fallback

    elif mode == "power":
        log("Albedo: fetching NASA POWER surface albedo (this may take a moment on first run)...")
        # Add project dir to path so power_albedo.py is importable
        sys.path.insert(0, str(Path(__file__).parent))
        from power_albedo import build_power_rho_series
        lla_df = pd.read_csv(f"{dd}/ISS_Model_LLA_Position.csv")
        lla_df["t"] = pd.to_datetime(lla_df["Time (UTCG)"], format=STK_TIME_FMT)
        rho_series = build_power_rho_series(
            lla_df,
            cache_dir=config.get("power_cache_dir", DEFAULT_CACHE_DIR),
            parameter=config.get("power_param", "ALLSKY_SRF_ALB"),
            fallback=fallback,
        )
        df = pd.merge_asof(
            df.sort_values("t"),
            rho_series.sort_values("t"),
            on="t", direction="nearest",
        )
        df["rho_eff"] = df["rho_eff"].fillna(fallback)

    elif mode == "ceres":
        ceres_path = f"{dd}/ceres_albedo_1day.csv"
        log(f"Albedo: loading CERES CSV ({ceres_path})...")
        ceres_df = pd.read_csv(ceres_path)
        ceres_df["t"] = pd.to_datetime(ceres_df["t"])
        rho_series = ceres_df[["t", "rho_eff"]].sort_values("t")
        df = pd.merge_asof(
            df.sort_values("t"),
            rho_series,
            on="t", direction="nearest",
        )
        df["rho_eff"] = df["rho_eff"].fillna(fallback)

    else:
        raise ValueError(f"Unknown albedo_mode: {mode!r}")

    # ---- LVLH frame + Sun direction ------------------------------------
    log("Computing LVLH frame and Sun direction...")
    r     = df[["r_x", "r_y", "r_z"]].to_numpy()
    v     = df[["v_x", "v_y", "v_z"]].to_numpy()
    Rhat  = _unit(r)
    Hhat  = _unit(np.cross(r, v))
    That  = _unit(np.cross(Hhat, Rhat))
    s_hat = _unit(df[["sun_x", "sun_y", "sun_z"]].to_numpy() - r)

    s_R = np.sum(s_hat * Rhat, axis=1)
    s_T = np.sum(s_hat * That, axis=1)
    s_H = np.sum(s_hat * Hhat, axis=1)
    df["s_R"] = s_R;  df["s_T"] = s_T;  df["s_H"] = s_H

    # ---- Earth albedo geometry ----------------------------------------
    r_mag  = np.linalg.norm(r, axis=1)
    F_earth    = np.sin(np.arcsin(np.clip(R_EARTH_KM / r_mag, 0., 1.))) ** 2
    day_factor = np.clip(-s_R, 0., 1.)
    e_hat      = np.column_stack([-np.ones(len(df)), np.zeros(len(df)), np.zeros(len(df))])
    df["alt_km"]     = r_mag - R_EARTH_KM
    df["F_earth"]    = F_earth
    df["day_factor"] = day_factor

    # ---- Sun az/el in sensor frame ------------------------------------
    az_offset = config.get("az_offset_deg", 0.0)
    sun_el    = np.degrees(np.arcsin(np.clip(s_H, -1., 1.)))
    sun_az    = (np.degrees(np.arctan2(s_T, s_R)) + az_offset + 180.) % 360. - 180.
    df["sun_az_deg"] = sun_az
    df["sun_el_deg"] = sun_el

    # ---- Irradiance per face (unmasked) --------------------------------
    log("Computing per-face irradiance...")
    rho = df["rho_eff"].to_numpy()
    for face, n in FACE_NORMALS.items():
        cos_sun   = np.clip(n[0]*s_R + n[1]*s_T + n[2]*s_H,         0., 1.)
        cos_earth = np.clip(n[0]*e_hat[:,0] + n[1]*e_hat[:,1] + n[2]*e_hat[:,2], 0., 1.)
        df[f"I_direct_{face}"] = AM0 * cos_sun
        df[f"I_albedo_{face}"] = rho * AM0 * F_earth * day_factor * cos_earth
        df[f"I_{face}"]        = df[f"I_direct_{face}"] + df[f"I_albedo_{face}"]

    # ---- Eclipse factor -----------------------------------------------
    log("Parsing STK lighting report...")
    _, penumbra_df, umbra_df = parse_lighting_report(
        f"{dd}/ISS_Model_Lighting_Times.csv"
    )
    ef, states = build_eclipse_factor(df["t"].to_numpy(), umbra_df, penumbra_df)
    df["eclipse_factor"] = ef
    df["lighting_state"] = states
    df["sun_factor"]     = ef

    # ---- Structural factor -------------------------------------------
    log("Applying az/el structural mask...")
    excl_zones, incl_map = parse_az_el_mask(f"{dd}/MSC_Sun_Sensor_Az_El_Mask.csv")
    sf, blocked = build_structural_factor(sun_az, sun_el, excl_zones, incl_map)
    df["structural_factor"]  = sf
    df["structural_blocked"] = blocked

    # ---- Masked irradiance + ESH ------------------------------------
    log("Computing masked irradiance and ESH...")
    esh_results = {}
    _ef  = df["eclipse_factor"].to_numpy()
    _sf  = df["structural_factor"].to_numpy()
    _xf  = df["exposure_open_factor"].to_numpy()
    dt_s = df["dt"].to_numpy()

    for face in FACE_NORMALS:
        Id = df[f"I_direct_{face}"].to_numpy()
        Ia = df[f"I_albedo_{face}"].to_numpy()
        Im_d = Id * _ef * _sf * _xf
        Im_a = Ia * _ef * _xf
        Im   = Im_d + Im_a
        df[f"I_direct_masked_{face}"] = Im_d
        df[f"I_albedo_masked_{face}"] = Im_a
        df[f"I_total_masked_{face}"]  = Im
        df[f"I_{face}"]               = Im    # downstream compatibility
        esh_results[face] = np.sum(Im * dt_s) / AM0 / 3600.

    # ---- Cumulative ESH ----------------------------------------------
    for face in FACE_NORMALS:
        cum = np.cumsum(df[f"I_total_masked_{face}"].to_numpy() * dt_s)
        df[f"ESH_{face}_cum"] = cum / AM0 / 3600.

    # ---- ESH summary table ------------------------------------------
    esh_summary = (
        pd.Series(esh_results, name="ESH (hours)")
        .sort_values(ascending=False)
        .to_frame()
    )
    esh_summary.index.name = "Face"
    esh_summary["ESH (hours)"] = esh_summary["ESH (hours)"].round(4)

    return {
        "df":         df,
        "esh_results": esh_results,
        "esh_summary": esh_summary,
        "export_df":  _build_export_df(df, config["primary_face"]),
        "excl_zones": excl_zones,
        "incl_map":   incl_map,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _fig_irradiance(df, face):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t = df["t"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Panel 1 — eclipse factor
    ax = axes[0]
    ax.fill_between(t, df["eclipse_factor"], alpha=0.3, color="gold")
    ax.plot(t, df["eclipse_factor"], color="goldenrod", lw=0.8, label="eclipse_factor")
    ax.set_ylabel("Eclipse\nfactor")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2 — direct + albedo irradiance
    ax = axes[1]
    dc, ac = f"I_direct_masked_{face}", f"I_albedo_masked_{face}"
    if dc in df.columns:
        ax.fill_between(t, 0, df[dc],           alpha=0.65, color="orange",  label="Direct")
        ax.fill_between(t, df[dc], df[dc] + df[ac], alpha=0.5,  color="skyblue", label="Albedo (POWER)")
    else:
        ax.fill_between(t, 0, df[f"I_{face}"], alpha=0.65, color="orange", label="Total")
    ax.set_ylabel("Irradiance\n(W/m²)")
    ax.set_title(f"Face {face} — masked irradiance", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3 — cumulative ESH
    ax = axes[2]
    ax.plot(t, df[f"ESH_{face}_cum"], color="green", lw=1.4)
    ax.set_ylabel("Cumulative\nESH (h)")
    ax.set_xlabel("Time (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    return fig


def _fig_all_faces(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.cm.tab10
    for k, face in enumerate(FACE_NORMALS):
        col = f"ESH_{face}_cum"
        if col in df.columns:
            ax.plot(df["t"], df[col], label=face, color=cmap(k / 10))
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Cumulative ESH (h)")
    ax.set_title("Cumulative ESH — All Faces (eclipse + structural + exposure masked)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    return fig


def _fig_rho_eff(df):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(df["t"], df["rho_eff"], lw=0.8, color="steelblue")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("ρ_eff (albedo)")
    ax.set_title("Effective albedo ρ_eff(t) — NASA POWER ALLSKY_SRF_ALB along ISS groundtrack")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    return fig


def _fig_az_el(df, excl_zones, incl_map):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.Set1
    for i, zone in enumerate(excl_zones):
        poly = MplPolygon(
            zone, closed=True, fill=True, alpha=0.2,
            facecolor=cmap(i % 9), edgecolor=cmap(i % 9), lw=1.2,
        )
        ax.add_patch(poly)
        cx, cy = zone[:, 0].mean(), zone[:, 1].mean()
        ax.text(cx, cy, f"EZ{i+1}", fontsize=7, ha="center", va="center")
    for excl_i, zones in incl_map.items():
        for z in zones:
            poly = MplPolygon(
                z, closed=True, fill=True, alpha=0.5,
                facecolor="white", edgecolor="limegreen", linestyle="--", lw=1,
            )
            ax.add_patch(poly)
    ax.plot(df["sun_az_deg"], df["sun_el_deg"], ".", ms=1.2, alpha=0.35,
            color="navy", label="Sun track (1-min)")
    ax.set_xlabel("Sun Azimuth (deg)")
    ax.set_ylabel("Sun Elevation (deg)")
    ax.set_title("Az/El Diagnostic — Sun Track vs Structural Exclusion Mask")
    ax.legend(fontsize=8)
    ax.set_xlim(-181, 181)
    ax.set_ylim(-91, 91)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MSC ESH Simulation",
    page_icon=":sunny:",
    layout="wide",
)

st.title("MSC ESH Simulation")
st.caption(
    "Equivalent Sun Hours for the MISSE Science Carrier — 6-month ISS simulation"
)

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    data_dir = st.text_input(
        "Data directory",
        value=DEFAULT_DATA_DIR,
        help="Folder containing ISS STK CSV exports.",
    )

    primary_face = st.selectbox(
        "Primary face (export & plots)",
        VALID_FACES,
        index=VALID_FACES.index("+H"),
    )

    st.divider()
    st.subheader("Albedo model")

    albedo_mode = st.radio(
        "Mode",
        options=["power", "constant", "ceres"],
        index=0,
        help=(
            "**power** — NASA POWER daily surface albedo (Tier A, recommended).\n\n"
            "**constant** — fixed ρ everywhere (Tier C).\n\n"
            "**ceres** — prepared CERES SYN1deg CSV (Tier B)."
        ),
    )

    albedo_constant = st.number_input(
        "Constant albedo ρ (fallback)",
        value=0.27, min_value=0.0, max_value=1.0, step=0.01,
        help="Used directly in 'constant' mode, or as fallback in 'power' mode.",
    )

    power_param = "ALLSKY_SRF_ALB"
    power_cache = DEFAULT_CACHE_DIR
    if albedo_mode == "power":
        power_param = st.selectbox(
            "POWER parameter",
            ["ALLSKY_SRF_ALB", "CLRSKY_SRF_ALB"],
            help=(
                "**ALLSKY_SRF_ALB** — all-sky surface albedo (clouds included).\n\n"
                "**CLRSKY_SRF_ALB** — clear-sky surface albedo only."
            ),
        )
        power_cache = st.text_input(
            "POWER cache directory",
            value=DEFAULT_CACHE_DIR,
            help="JSON cache files are written here so repeated runs skip re-querying.",
        )

    st.divider()
    st.subheader("Exposure windows")
    st.caption(
        "Leave blank for full simulation period.  "
        "Format: `DD Mon YYYY HH:MM:SS.mmm | DD Mon YYYY HH:MM:SS.mmm` — one window per line."
    )
    exposure_text = st.text_area(
        "Exposure windows",
        value="",
        height=90,
        placeholder="04 Mar 2026 10:54:00.000 | 04 Mar 2026 11:54:00.000",
        label_visibility="collapsed",
    )

    st.divider()
    run_btn = st.button(
        "Run Simulation",
        type="primary",
        use_container_width=True,
    )

# ---- Main area -------------------------------------------------------------
if "result" not in st.session_state:
    st.info(
        "Configure the parameters in the sidebar and click **Run Simulation** to start.\n\n"
        "- With **power** mode selected the app will call the NASA POWER API on the first run "
        "and cache results locally — subsequent runs for the same date are instant.\n"
        "- The output CSV is compatible with the downstream `sensor_channel_split.ipynb` notebook."
    )
    st.stop()

# ---- Run when button pressed -----------------------------------------------
if run_btn:
    try:
        windows = parse_exposure_windows(exposure_text) if exposure_text.strip() else []
    except ValueError as exc:
        st.error(f"Exposure window error: {exc}")
        st.stop()

    cfg = {
        "data_dir":         data_dir,
        "primary_face":     primary_face,
        "exposure_windows": windows,
        "albedo_mode":      albedo_mode,
        "albedo_constant":  albedo_constant,
        "power_param":      power_param,
        "power_cache_dir":  power_cache,
        "az_offset_deg":    0.0,
    }

    log_lines: list = []
    status_placeholder = st.empty()

    def _log(msg):
        log_lines.append(f"• {msg}")
        status_placeholder.info("\n".join(log_lines))

    with st.spinner("Running simulation..."):
        try:
            result = run_simulation(cfg, _log)
            st.session_state["result"] = result
            st.session_state["cfg"]    = cfg
            status_placeholder.success(
                f"Done — primary face: **{primary_face}**  |  "
                f"ESH: **{result['esh_results'][primary_face]:.3f} h**  |  "
                f"albedo mode: **{albedo_mode}**"
            )
        except FileNotFoundError as exc:
            st.error(f"Data file not found: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Simulation error: {exc}")
            st.exception(exc)
            st.stop()

# ---- Display results -------------------------------------------------------
if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]
cfg    = st.session_state["cfg"]
df          = result["df"]
esh_summary = result["esh_summary"]
export_df   = result["export_df"]
excl_zones  = result["excl_zones"]
incl_map    = result["incl_map"]
face        = cfg["primary_face"]

# ---- Top row: summary + info -----------------------------------------------
col_esh, col_info = st.columns([1, 2])

with col_esh:
    st.subheader("ESH Summary")
    st.dataframe(esh_summary, use_container_width=True)

with col_info:
    sim_dur_h  = (df["t"].iloc[-1] - df["t"].iloc[0]).total_seconds() / 3600.
    n_umbra    = int((df["eclipse_factor"] == 0.0).sum())
    n_penu     = int((df["eclipse_factor"] == 0.5).sum())
    n_sun      = int((df["eclipse_factor"] == 1.0).sum())
    n_blocked  = int(df["structural_blocked"].sum())
    open_pct   = df["exposure_open_factor"].mean() * 100

    st.subheader("Simulation info")
    info_df = pd.DataFrame({
        "Parameter": [
            "Start", "End", "Duration (h)", "Timesteps",
            "Albedo mode", "ρ mean", "ρ min", "ρ max",
            "Umbra steps", "Penumbra steps", "Sunlight steps",
            "Struct. blocked steps", "Exposure open (%)",
        ],
        "Value": [
            str(df["t"].iloc[0])[:19],
            str(df["t"].iloc[-1])[:19],
            f"{sim_dur_h:.2f}",
            len(df),
            cfg["albedo_mode"],
            f"{df['rho_eff'].mean():.4f}",
            f"{df['rho_eff'].min():.4f}",
            f"{df['rho_eff'].max():.4f}",
            n_umbra, n_penu, n_sun, n_blocked,
            f"{open_pct:.1f}",
        ],
    }).set_index("Parameter")
    st.dataframe(info_df, use_container_width=True)

# ---- Tabs: plots -----------------------------------------------------------
st.divider()
tab_face, tab_all, tab_rho, tab_azel = st.tabs([
    f"Irradiance — {face}",
    "Cumulative ESH — all faces",
    "ρ_eff(t) — albedo track",
    "Az/El diagnostic",
])

with tab_face:
    st.pyplot(_fig_irradiance(df, face), use_container_width=True)

with tab_all:
    st.pyplot(_fig_all_faces(df), use_container_width=True)

with tab_rho:
    st.pyplot(_fig_rho_eff(df), use_container_width=True)

with tab_azel:
    st.pyplot(_fig_az_el(df, excl_zones, incl_map), use_container_width=True)

# ---- Downloads -------------------------------------------------------------
st.divider()
dl1, dl2 = st.columns(2)

with dl1:
    safe_face = face.replace("+", "p").replace("-", "n")
    st.download_button(
        label=f"Download irradiance CSV — face {face}",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"msc_esh_{safe_face}_irradiance.csv",
        mime="text/csv",
        use_container_width=True,
    )

with dl2:
    st.download_button(
        label="Download full simulation DataFrame",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="msc_esh_full_simulation.csv",
        mime="text/csv",
        use_container_width=True,
    )
