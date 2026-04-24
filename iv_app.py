import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from math import ceil
from pathlib import Path
from PIL import Image
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="I-V Curve Analysis", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .status-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-top: 4px;
    }
    .badge-normal  { background: #27ae60; color: white; }
    .badge-warning { background: #f1c40f; color: black; }
    .badge-stepped { background: #e74c3c; color: white; }
    .badge-nodata  { background: #95a5a6; color: white; }
</style>
""", unsafe_allow_html=True)

# --- CONFIG PATHS ---
PARQUET_PATH = Path(r"C:\Users\alwi_fahrozi\OneDrive - Banpu Public Company Limited\Operation & Maintenance - IV_CURVE\scraping_iv_curve\iv_app_repo\iv-database-1.parquet")
LOGO_PATH    = r"C:\Users\alwi_fahrozi\OneDrive - Banpu Public Company Limited\Operation & Maintenance - IV_CURVE\scraping_iv_curve\iv_app_repo\logo cpi.jpg"

# --- RENDERER ---
def render_status_text(severity, message):
    color_map = {
        "Normal":  "#27ae60",
        "Warning": "#f1c40f",
        "Stepped": "#e74c3c",
        "No Data": "#95a5a6",
    }
    bg_map = {
        "Normal":  "rgba(39, 174, 96, 0.12)",
        "Warning": "rgba(241, 196, 15, 0.18)",
        "Stepped": "rgba(231, 76, 60, 0.12)",
        "No Data": "rgba(149, 165, 166, 0.15)",
    }
    color = color_map.get(severity, "#000000")
    bg    = bg_map.get(severity, "#f5f5f5")
    st.markdown(f"""
    <div style="background-color:{bg}; padding:12px 16px; border-radius:10px;
                border-left:6px solid {color}; font-size:14px;">
        <span style="color:{color}; font-weight:bold;">status:</span>
        <span style="color:{color};"> {message}</span>
    </div>""", unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def load_data(path):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
    for c in ["data_point", "Voltage_V", "Current_A"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Site_Name"]     = df["Site_Name"].astype(str).str.strip()
    df["Serial_Number"] = df["Serial_Number"].astype(str).str.strip()
    df["PV_Input"]      = df["PV_Input"].astype(str)
    df["Test_Time"]     = pd.to_datetime(df["Test_Time"], errors="coerce")
    df["Date"]          = df["Test_Time"].dt.date
    return df.dropna(subset=["Site_Name", "Serial_Number", "Test_Time"])

df_full = load_data(PARQUET_PATH)
if df_full is None:
    st.error(f"Database file not found: {PARQUET_PATH}")
    st.stop()

# ============================================================
# COLLECT RAW FEATURES FROM ALL PV STRINGS
# ============================================================

@st.cache_data
def build_global_pool(df):
    slope_vars, max_slope_drops, peak_accels = [], [], []
    v_ranges, n_points_list = [], []

    group_cols = ["Test_Time", "Site_Name", "Serial_Number", "PV_Input"]

    for key, sub in df.groupby(group_cols, observed=True):
        sub = (sub.dropna(subset=["Voltage_V", "Current_A"])
                  .sort_values("Voltage_V"))
        v     = sub["Voltage_V"].values
        i_raw = sub["Current_A"].values

        if len(v) < 8:
            continue
        v_range = float(v[-1] - v[0])
        if v_range < 1.0:
            continue

        n      = len(i_raw)
        window = max(5, min(int(n / 10), 21))
        if window % 2 == 0:
            window += 1

        try:
            i_sm = savgol_filter(i_raw, window_length=window, polyorder=2)
        except Exception:
            i_sm = i_raw.copy()

        slope = np.gradient(i_sm, v)
        accel = np.gradient(slope, v)

        slope_vars.append(float(np.var(slope)))
        sd = np.diff(slope)
        max_slope_drops.append(float(np.max(-sd)) if len(sd) else 0.0)

        trim = max(2, int(0.05 * len(accel)))
        core = accel[trim:-trim]
        if len(core) > 5:
            mad   = np.median(np.abs(core - np.median(core)))
            z_acc = (accel - np.median(core)) / (1.4826 * mad + 1e-9)
            peak_accels.append(float(np.max(np.abs(z_acc))))
        else:
            peak_accels.append(0.0)

        v_ranges.append(v_range)
        n_points_list.append(n)

    if not slope_vars:
        return None

    return {
        "slope_vars":      np.array(slope_vars),
        "max_slope_drops": np.array(max_slope_drops),
        "peak_accels":     np.array(peak_accels),
        "v_ranges":        np.array(v_ranges),
        "n_points":        np.array(n_points_list),
        "n_total":         len(slope_vars),
    }

global_pool = build_global_pool(df_full)

if global_pool is None:
    st.error("Tidak ada data valid di database. Periksa file parquet.")
    st.stop()

# ============================================================
# LEARN ALL PARAMETERS FROM GLOBAL POOL
# ============================================================

@st.cache_data
def learn_parameters_from_global_pool(_pool):
    drops  = _pool["max_slope_drops"]
    accels = _pool["peak_accels"]
    vars_  = _pool["slope_vars"]
    vrng   = _pool["v_ranges"]
    n_pts  = _pool["n_points"]
    N      = _pool["n_total"]

    n_window = int(np.clip(np.percentile(np.arange(1, N + 1), 80), 50, N))
    tier_full_min    = max(30,  int(np.ceil(0.010 * N)))
    tier_partial_min = max(10,  int(np.ceil(0.003 * N)))

    fallback_slope_change = float(np.median(drops))
    fallback_accel_z      = float(np.median(accels))
    fallback_pen_factor   = 1.0

    cv_drops      = float(np.std(drops) / (np.mean(drops) + 1e-9))
    partial_damping = float(np.clip(0.3 + 0.4 * cv_drops, 0.3, 0.7))

    sorted_drops = np.sort(drops)
    cdf          = np.arange(1, N + 1) / N
    if N > 20:
        d2    = np.gradient(np.gradient(cdf, sorted_drops + 1e-9), sorted_drops + 1e-9)
        elbow = int(np.argmax(np.abs(d2[5:-5])) + 5)
        slope_change_pct = float(np.clip(cdf[elbow] * 100, 75, 95))
    else:
        slope_change_pct = 85.0

    sorted_accels = np.sort(accels)
    if N > 20:
        d2_a    = np.gradient(np.gradient(cdf, sorted_accels + 1e-9), sorted_accels + 1e-9)
        elbow_a = int(np.argmax(np.abs(d2_a[5:-5])) + 5)
        accel_z_pct = float(np.clip(cdf[elbow_a] * 100, 88, 97))
    else:
        accel_z_pct = 94.0

    p75_d, p95_d = np.percentile(drops, [75, 95])
    iqr_d         = p75_d - np.percentile(drops, 25)
    if iqr_d > 1e-9:
        tail_ratio    = (p95_d - p75_d) / iqr_d
        iqr_cap_mult  = float(np.clip(tail_ratio, 1.0, 3.0))
    else:
        iqr_cap_mult = 1.5

    resolution   = float(np.median(vrng / (n_pts + 1e-9)))
    median_vrange= float(np.median(vrng))
    merge_v_frac = float(np.clip(resolution / (median_vrange + 1e-9) * 3, 0.02, 0.08))

    return {
        "n_window":               n_window,
        "tier_full_min":          tier_full_min,
        "tier_partial_min":       tier_partial_min,
        "fallback_slope_change": fallback_slope_change,
        "fallback_accel_z":       fallback_accel_z,
        "fallback_pen_factor": fallback_pen_factor,
        "partial_damping":     partial_damping,
        "slope_change_pct":     slope_change_pct,
        "accel_z_pct":          accel_z_pct,
        "iqr_cap_mult":         iqr_cap_mult,
        "merge_v_frac":         merge_v_frac,
        "n_total":             N,
        "median_v_range":       median_vrange,
    }

LEARNED = learn_parameters_from_global_pool(global_pool)

HISTORICAL_WINDOW       = LEARNED["n_window"]
TIER_FULL_MIN           = LEARNED["tier_full_min"]
TIER_PARTIAL_MIN        = LEARNED["tier_partial_min"]
MIN_SAMPLES_REQUIRED    = TIER_PARTIAL_MIN
FALLBACK_SLOPE_CHANGE   = LEARNED["fallback_slope_change"]
FALLBACK_ACCEL_Z        = LEARNED["fallback_accel_z"]
FALLBACK_PEN_FACTOR     = LEARNED["fallback_pen_factor"]
PARTIAL_DAMPING         = LEARNED["partial_damping"]
SLOPE_CHANGE_PERCENTILE = LEARNED["slope_change_pct"]
ACCEL_Z_PERCENTILE      = LEARNED["accel_z_pct"]
IQR_CAP_MULTIPLIER      = LEARNED["iqr_cap_mult"]
MERGE_VOLTAGE_FRACTION  = LEARNED["merge_v_frac"]

# ============================================================
# ADAPTIVE THRESHOLDS DARI GLOBAL POOL
# ============================================================

def _iqr_robust_percentile(arr, pct):
    q25, q75 = np.percentile(arr, [25, 75])
    iqr       = q75 - q25
    clipped   = arr[arr <= q75 + IQR_CAP_MULTIPLIER * iqr]
    if len(clipped) < 3:
        clipped = arr
    return float(np.percentile(clipped, pct))

def get_adaptive_thresholds():
    base = {
        "min_slope_change": FALLBACK_SLOPE_CHANGE,
        "accel_z_thresh":   FALLBACK_ACCEL_Z,
        "pen_factor":       FALLBACK_PEN_FACTOR,
        "tier":             "fallback",
        "n_obs":            global_pool["n_total"],
        "mean_v_range":     LEARNED["median_v_range"],
    }

    N = global_pool["n_total"]
    if N < TIER_PARTIAL_MIN:
        return base

    n_use         = min(HISTORICAL_WINDOW, N)
    recent_drops  = global_pool["max_slope_drops"][-n_use:]
    recent_accels = global_pool["peak_accels"][-n_use:]
    recent_vars   = global_pool["slope_vars"][-n_use:]

    fa_slope = max(_iqr_robust_percentile(recent_drops,  SLOPE_CHANGE_PERCENTILE), FALLBACK_SLOPE_CHANGE * 0.5)
    fa_accel = float(np.clip(
        _iqr_robust_percentile(recent_accels, ACCEL_Z_PERCENTILE),
        FALLBACK_ACCEL_Z * 0.7,
        FALLBACK_ACCEL_Z * 2.2,
    ))

    med_var = np.median(recent_vars)
    mad_var = np.median(np.abs(recent_vars - med_var)) + 1e-9
    pen     = float(np.clip(1.0 + 0.3 * (med_var / (mad_var + 1e-9) - 1.0), 0.4, 2.5))

    if N >= TIER_FULL_MIN:
        return {
            "min_slope_change": fa_slope,
            "accel_z_thresh":   fa_accel,
            "pen_factor":       pen,
            "tier":             "full",
            "n_obs":            N,
            "mean_v_range":     LEARNED["median_v_range"],
        }
    else:
        alpha = PARTIAL_DAMPING * (N - TIER_PARTIAL_MIN) / max(TIER_FULL_MIN - TIER_PARTIAL_MIN, 1)
        alpha = float(np.clip(alpha, 0.0, PARTIAL_DAMPING))
        return {
            "min_slope_change": (1 - alpha) * FALLBACK_SLOPE_CHANGE + alpha * fa_slope,
            "accel_z_thresh":   (1 - alpha) * FALLBACK_ACCEL_Z      + alpha * fa_accel,
            "pen_factor":       (1 - alpha) * FALLBACK_PEN_FACTOR   + alpha * pen,
            "tier":             "partial",
            "n_obs":            N,
            "mean_v_range":     LEARNED["median_v_range"],
        }

# ============================================================
# STEP DETECTION & FEATURE EXTRACTION & HYBRID
# ============================================================
# (Fungsi-fungsi pendukung tetap sama)

def detect_steps_self_contained(v, slope, z_accel, min_slope_change, accel_z_thresh, merge_radius):
    n          = len(slope)
    if n < 10: return []
    slope_diff = np.abs(np.diff(slope))
    candidates = np.where(slope_diff > min_slope_change)[0]
    steps = []
    for idx in candidates:
        if idx < 2 or idx > n - 3: continue
        lo, hi = max(0, idx - 1), min(n, idx + 3)
        if np.max(np.abs(z_accel[lo:hi])) > accel_z_thresh:
            steps.append(float((v[idx] + v[min(idx + 1, n - 1)]) / 2.0))
    if not steps: return []
    steps.sort()
    merged = [steps[0]]
    for s in steps[1:]:
        if s - merged[-1] > merge_radius: merged.append(s)
    return merged

def binary_segmentation_change_points(signal, min_seg_len=4, pen_factor=1.0):
    n = len(signal)
    if n < 2 * min_seg_len: return []
    sig_var = np.var(signal)
    if sig_var < 1e-12: return []
    penalty = pen_factor * np.log(max(n, 2)) * sig_var
    def _cost(seg): return float(np.sum((seg - np.mean(seg)) ** 2))
    def _best_split(start, end):
        if end - start < 2 * min_seg_len: return None, 0.0
        base = _cost(signal[start:end])
        best_idx, best_r = None, 0.0
        for i in range(start + min_seg_len, end - min_seg_len + 1):
            r = base - _cost(signal[start:i]) - _cost(signal[i:end])
            if r > best_r: best_r, best_idx = r, i
        return best_idx, best_r
    cps, segments = [], [(0, n)]
    while segments:
        s, e = segments.pop(0)
        idx, r = _best_split(s, e)
        if idx is not None and r > penalty:
            cps.append(idx)
            segments += [(s, idx), (idx, e)]
    return sorted(cps)

def extract_gradient_features(v, i):
    if len(v) < 8 or np.all(i == 0): return None
    n = len(i)
    window = max(5, min(int(n / 10), 25))
    if window % 2 == 0: window += 1
    try: i_sm = savgol_filter(i, window_length=window, polyorder=2)
    except: i_sm = i.copy()
    i_max = float(np.max(i_sm))
    v_range = float(v[-1] - v[0])
    if i_max <= 0 or v_range < 1.0: return None
    slope = np.gradient(i_sm, v)
    accel = np.gradient(slope, v)
    slope_var = float(np.var(slope))
    sd = np.diff(slope)
    max_slope_drop = float(np.max(-sd)) if len(sd) else 0.0
    trim = max(2, int(0.05 * len(accel)))
    core = accel[trim:-trim]
    if len(core) > 5:
        mad = np.median(np.abs(core - np.median(core)))
        z_accel = (accel - np.median(core)) / (1.4826 * mad + 1e-9)
        peak_accel = float(np.max(np.abs(z_accel)))
    else:
        z_accel = np.zeros_like(accel); peak_accel = 0.0
    hist_z_slope_drop = hist_z_peak_accel = 0.0
    n_use = min(HISTORICAL_WINDOW, global_pool["n_total"])
    if n_use >= TIER_PARTIAL_MIN:
        r_drops = global_pool["max_slope_drops"][-n_use:]; r_peaks = global_pool["peak_accels"][-n_use:]
        hist_z_slope_drop = (max_slope_drop - np.mean(r_drops)) / (np.std(r_drops) + 1e-9)
        hist_z_peak_accel = (peak_accel - np.mean(r_peaks)) / (np.std(r_peaks) + 1e-9)
    return {"v": v, "i": i_sm, "i_max": i_max, "v_range": v_range, "slope": slope, "accel": accel, "z_accel": z_accel, "slope_var": slope_var, "max_slope_drop": max_slope_drop, "peak_accel": peak_accel, "hist_z_slope_drop": hist_z_slope_drop, "hist_z_peak_accel": hist_z_peak_accel, "feat_vector": [slope_var, max_slope_drop, peak_accel, hist_z_slope_drop, hist_z_peak_accel]}

def detect_stepped_curves_hybrid(df_current):
    results = {}
    thr = get_adaptive_thresholds()
    min_sc = thr["min_slope_change"]; accel_z = thr["accel_z_thresh"]; pen = thr["pen_factor"]; tier = thr["tier"]; mean_vr = thr["mean_v_range"]
    is_adaptive = tier in ("full", "partial")
    pvs = df_current["PV_Input"].unique(); feat_matrix = []; valid_pvs = []; temp_data = {}
    for p in pvs:
        sub = (df_current[df_current["PV_Input"] == p].dropna(subset=["Voltage_V", "Current_A"]).sort_values("Voltage_V"))
        feat = extract_gradient_features(sub["Voltage_V"].values, sub["Current_A"].values)
        if feat:
            feat_matrix.append(feat["feat_vector"]); valid_pvs.append(p); temp_data[p] = feat
        else:
            results[p] = {"severity": "No Data", "details": "Insufficient data points.", "step_positions": [], "slope_profile_v": [], "slope_profile_s": []}
    if not feat_matrix: return results, min_sc, accel_z, is_adaptive
    use_ml = len(feat_matrix) >= 6; ml_preds = np.ones(len(valid_pvs), dtype=int)
    if use_ml:
        X = StandardScaler().fit_transform(np.array(feat_matrix))
        ml_preds = IsolationForest(contamination="auto", random_state=42, n_estimators=120).fit_predict(X)
    for idx, p in enumerate(valid_pvs):
        feat = temp_data[p]; ml_anom = bool(ml_preds[idx] == -1) if use_ml else False
        merge_r = max(2.0, MERGE_VOLTAGE_FRACTION * (feat["v_range"] if feat["v_range"] > 0 else mean_vr))
        min_seg = max(3, int(0.015 * len(feat["v"])))
        seg_steps = [float(feat["v"][i]) for i in binary_segmentation_change_points(feat["slope"], min_seg_len=min_seg, pen_factor=pen) if 0 <= i < len(feat["v"])]
        sc_steps = detect_steps_self_contained(feat["v"], feat["slope"], feat["z_accel"], min_sc, accel_z, merge_r)
        all_cands = sorted(set(seg_steps + sc_steps)); validated = []
        for pos in all_cands:
            i_v = int(np.argmin(np.abs(feat["v"] - pos)))
            delta = float(np.max(np.abs(np.diff(feat["slope"][max(0, i_v - 1):min(len(feat["slope"]) - 1, i_v + 1) + 2]))))
            accel_ok = float(np.max(np.abs(feat["z_accel"][max(0, i_v - 1):i_v + 2]))) > accel_z
            if accel_ok and delta > min_sc: validated.append(pos)
        if validated:
            validated.sort(); merged = [validated[0]]
            for s in validated[1:]:
                if s - merged[-1] > merge_r: merged.append(s)
            validated = merged
        is_hist_anom = is_adaptive and (abs(feat["hist_z_slope_drop"]) > 2.2 or abs(feat["hist_z_peak_accel"]) > 2.2)
        severity = "Stepped" if (len(validated) > 0 and (is_hist_anom or ml_anom)) else ("Warning" if len(validated) > 0 else "Normal")
        msg = f"Step detected: {len(validated)} points. Δslope: {min_sc:.4f} A/V, Z-accel: {accel_z:.2f}"
        if not is_adaptive: msg += f" | Fallback (obs global < {MIN_SAMPLES_REQUIRED})"
        results[p] = {"severity": severity, "details": msg, "step_positions": validated, "slope_profile_v": feat["v"].tolist(), "slope_profile_s": feat["slope"].tolist()}
    return results, min_sc, accel_z, is_adaptive

# ============================================================
# UI (MODIFIED FILTER HIERARCHY)
# ============================================================

def _pv_sort_key(pv):
    m = re.search(r"\d+", str(pv))
    return (int(m.group(0)) if m else 0, str(pv))

# --- SIDEBAR (Updated Hierarchy) ---
st.sidebar.header("📅 Data Filter")

# 1. SITE NAME FIRST
site_list = sorted(df_full["Site_Name"].unique())
site_selected = st.sidebar.selectbox("Site:", site_list)
df_by_site = df_full[df_full["Site_Name"] == site_selected]

# 2. SERIAL NUMBER SECOND
serial_list = sorted(df_by_site["Serial_Number"].unique())
serial_selected = st.sidebar.selectbox("Serial Number:", serial_list)
df_by_site_sn = df_by_site[df_by_site["Serial_Number"] == serial_selected]

# 3. DATE THIRD
date_list = sorted(df_by_site_sn["Date"].unique())
date_selected = st.sidebar.selectbox("Date:", date_list)

# Final selection for the main content
dff = df_full[
    (df_full["Date"]          == date_selected) &
    (df_full["Site_Name"]     == site_selected) &
    (df_full["Serial_Number"] == serial_selected)
].copy()

# --- MAIN PAGE ---
try:
    img_logo = Image.open(LOGO_PATH)
    head_col1, head_col2 = st.columns([1, 10])
    with head_col1:
        st.image(img_logo, width=80)
    with head_col2:
        st.title("I-V Curve Analysis")
except Exception:
    st.title("I-V Curve Analysis")


if not dff.empty:
    tab1, tab2, tab3 = st.tabs(["📈 Combined View", "🔲 Grid View", "🔬 Analysis"])

    with tab1:
        pvs      = sorted(dff["PV_Input"].unique(), key=_pv_sort_key)
        fig_comb = go.Figure()
        for pv in pvs:
            chunk = (dff[dff["PV_Input"] == pv]
                     .dropna(subset=["Voltage_V", "Current_A"])
                     .sort_values("Voltage_V"))
            if len(chunk) >= 5:
                fig_comb.add_trace(go.Scatter(
                    x=chunk["Voltage_V"], y=chunk["Current_A"],
                    mode="lines+markers", marker=dict(size=4),
                    name=f"PV {pv}",
                ))
        fig_comb.update_layout(
            xaxis_title="Voltage (V)", yaxis_title="Current (A)",
            template="plotly_white", height=500,
        )
        st.plotly_chart(fig_comb, use_container_width=True)

    with tab2:
        pvs      = sorted(dff["PV_Input"].unique(), key=_pv_sort_key)
        cols     = 4
        fig_grid = make_subplots(
            rows=ceil(len(pvs) / cols), cols=cols,
            subplot_titles=[f"PV: {pv}" for pv in pvs],
        )
        for i, pv in enumerate(pvs):
            r, c  = (i // cols) + 1, (i % cols) + 1
            chunk = (dff[dff["PV_Input"] == pv]
                     .dropna(subset=["Voltage_V", "Current_A"])
                     .sort_values("Voltage_V"))
            if len(chunk) >= 5:
                fig_grid.add_trace(go.Scatter(
                    x=chunk["Voltage_V"], y=chunk["Current_A"],
                    mode="lines+markers", marker=dict(size=3),
                    showlegend=False,
                ), row=r, col=c)
        fig_grid.update_layout(
            height=250 * ceil(len(pvs) / cols),
            template="plotly_white",
        )
        st.plotly_chart(fig_grid, use_container_width=True)

    with tab3:
        st.subheader("🔬 Step Detection")

        with st.spinner("Menganalisis kurva dengan parameter hasil pembelajaran global…"):
            analysis_results, used_slope, used_accel, is_adaptive = detect_stepped_curves_hybrid(dff)

        count_normal  = sum(1 for r in analysis_results.values() if r["severity"] == "Normal")
        count_warning = sum(1 for r in analysis_results.values() if r["severity"] == "Warning")
        count_stepped = sum(1 for r in analysis_results.values() if r["severity"] == "Stepped")
        count_nodata  = sum(1 for r in analysis_results.values() if r["severity"] == "No Data")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="🟢 Normal", value=count_normal)
        col2.metric(label="🟡 Warning", value=count_warning)
        col3.metric(label="🔴 Stepped", value=count_stepped)
        col4.metric(label="⚪ No Data", value=count_nodata)

        st.write("")

        # ── Status Heatmap ────────────────────────
        pvs_all    = sorted(dff["PV_Input"].unique(), key=_pv_sort_key)
        grid_cols  = 6
        status_num = {"Normal": 0, "Warning": 1, "Stepped": 2, "No Data": 3}
        colors     = ["#27ae60", "#f1c40f", "#e74c3c", "#95a5a6"]

        z_vals, text_vals, hover_vals = [], [], []
        for p in pvs_all:
            res = analysis_results.get(p, {"severity": "No Data", "details": ""})
            z_vals.append(status_num[res["severity"]])
            text_vals.append(f"<b>{p}</b>")
            hover_vals.append(f"PV: {p}<br>Status: {res['severity']}<br>{res['details']}")

        pad  = ceil(len(pvs_all) / grid_cols) * grid_cols - len(pvs_all)
        z_2d = np.array(z_vals     + [None] * pad, dtype=object).reshape(-1, grid_cols)
        t_2d = np.array(text_vals  + [""]   * pad).reshape(-1, grid_cols)
        h_2d = np.array(hover_vals + [""]   * pad).reshape(-1, grid_cols)

        fig_status = go.Figure(data=go.Heatmap(
            z=z_2d, text=t_2d, texttemplate="%{text}",
            hoverinfo="text", hovertext=h_2d,
            colorscale=[
                [0,    colors[0]], [0.25, colors[0]],
                [0.25, colors[1]], [0.5,  colors[1]],
                [0.5,  colors[2]], [0.75, colors[2]],
                [0.75, colors[3]], [1.0,  colors[3]],
            ],
            zmin=0, zmax=3, showscale=False, xgap=5, ygap=5,
        ))
        fig_status.update_layout(
            height=130 * z_2d.shape[0],
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange="reversed"),
            template="plotly_white",
        )
        st.plotly_chart(fig_status, use_container_width=True)


        # ── Detail per PV ─────────────────────────
        st.divider()
        sel = st.selectbox("Detailed AI Diagnostics:", pvs_all)
        if sel in analysis_results:
            res = analysis_results[sel]
            render_status_text(res["severity"], res["details"])

            if res["slope_profile_v"]:
                chunk   = dff[dff["PV_Input"] == sel].sort_values("Voltage_V")
                det_fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=["IV Curve", "Gradient Analysis"],
                )
                det_fig.add_trace(go.Scatter(
                    x=chunk["Voltage_V"], y=chunk["Current_A"],
                    mode="lines+markers", name="Current",
                ), row=1, col=1)
                det_fig.add_trace(go.Scatter(
                    x=res["slope_profile_v"], y=res["slope_profile_s"],
                    mode="lines", name="Gradient",
                ), row=2, col=1)
                for pos in res["step_positions"]:
                    det_fig.add_vline(x=pos, line_dash="dash", line_color="red", row=1, col=1)
                    det_fig.add_vline(x=pos, line_dash="dash", line_color="red", row=2, col=1)
                det_fig.update_layout(height=600, template="plotly_white", showlegend=False)
                st.plotly_chart(det_fig, use_container_width=True)
else:
    st.info("Please select filter to start AI Analysis.")

st.markdown("---")
st.caption(
    f"CPI Solar AI Engine | {site_selected} | "
    f" Learning Mode — {LEARNED['n_total']:,} observasi string PV"
)