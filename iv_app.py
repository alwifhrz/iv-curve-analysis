import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from math import ceil, sqrt
from pathlib import Path
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="I-V Curve Explorer", layout="wide")

# --- LOAD DATA ---
PARQUET_PATH = Path("iv-database-1.parquet")
LOGO_PATH = "logo cpi.jpg"

@st.cache_data
def load_data(path):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
    
    for c in ["data_point", "Voltage_V", "Current_A"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df["Site_Name"] = df["Site_Name"].astype(str).str.strip()
    df["Serial_Number"] = df["Serial_Number"].astype(str).str.strip()
    df["PV_Input"] = df["PV_Input"].astype(str)
    df["Test_Time"] = pd.to_datetime(df["Test_Time"], errors="coerce")
    df["Date"] = df["Test_Time"].dt.date
    
    return df.dropna(subset=["Site_Name", "Serial_Number", "Test_Time", "Voltage_V", "Current_A"])

df = load_data(PARQUET_PATH)

if df is None:
    st.error(f"File database tidak ditemukan: {PARQUET_PATH}")
    st.stop()

# --- SIDEBAR FILTERING ---
st.sidebar.header("📊 Filter Site")
site_list = sorted(df["Site_Name"].unique())
site_selected = st.sidebar.selectbox("Site:", site_list)

serial_list = sorted(df[df["Site_Name"] == site_selected]["Serial_Number"].unique())
serial_selected = st.sidebar.selectbox("Serial Number:", serial_list)

date_list = sorted(df[(df["Site_Name"] == site_selected) & 
                    (df["Serial_Number"] == serial_selected)]["Date"].unique())
date_selected = st.sidebar.selectbox("Date:", date_list)

# --- FILTER DATA ---
dff = df[
    (df["Site_Name"] == site_selected) & 
    (df["Serial_Number"] == serial_selected) & 
    (df["Date"] == date_selected)
].copy()

# --- HEADER ---
try:
    img_logo = Image.open(LOGO_PATH)
    head_col1, head_col2 = st.columns([1, 10])
    with head_col1:
        st.image(img_logo, width=80)
    with head_col2:
        st.title("I-V Curve Analysis")
except:
    st.title("I-V Curve Analysis")

# --- METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("PV Inputs", dff["PV_Input"].nunique())
col2.metric("Max Voltage", f"{round(dff['Voltage_V'].max(), 2)} V")
col3.metric("Max Current", f"{round(dff['Current_A'].max(), 2)} A")

st.divider()

# --- PLOT HELPERS (PLOTLY) ---
def _pv_sort_key(pv):
    s = str(pv).strip().upper()
    m = re.search(r"\d+", s)
    return (int(m.group(0)) if m else 10**9, s)

def render_plotly_combined(df_file):
    fig = go.Figure()
    pvs = sorted(df_file["PV_Input"].unique(), key=_pv_sort_key)
    
    for pv in pvs:
        chunk = df_file[df_file["PV_Input"] == pv].sort_values(["Voltage_V", "data_point"])
        fig.add_trace(go.Scatter(
            x=chunk["Voltage_V"], 
            y=chunk["Current_A"],
            mode='lines+markers',
            name=f"PV {pv}",
            marker=dict(size=4),
            line=dict(width=2),
            hovertemplate="Voltage: %{x}V<br>Current: %{y}A<extra></extra>"
        ))

    fig.update_layout(
        # title=f"Combined IV Curves | {serial_selected}",
        xaxis_title="Voltage (V)",
        yaxis_title="Current (A)",
        legend_title="PV Input",
        hovermode="closest",
        template="plotly_white",
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def render_plotly_grid(df_file):
    pvs = sorted(df_file["PV_Input"].unique(), key=_pv_sort_key)
    n = len(pvs)
    cols = 3 
    rows = ceil(n / cols)

    fig = make_subplots(
        rows=rows, cols=cols, 
        subplot_titles=[f"PV: {pv}" for pv in pvs],
        shared_xaxes=False,
        shared_yaxes=False
    )

    for i, pv in enumerate(pvs):
        r = (i // cols) + 1
        c = (i % cols) + 1
        chunk = df_file[df_file["PV_Input"] == pv].sort_values(["Voltage_V", "data_point"])
        
        fig.add_trace(
            go.Scatter(
                x=chunk["Voltage_V"], 
                y=chunk["Current_A"],
                mode='lines+markers',
                marker=dict(size=3, color="#2c3e50"),
                name=f"PV {pv}",
                showlegend=False
            ),
            row=r, col=c
        )

    fig.update_layout(
        height=300 * rows, 
        template="plotly_white",
        title_text="IV Curves per PV Input Grid",
        margin=dict(t=80, b=20)
    )
    fig.update_xaxes(title_text="V", row=rows, col=1)
    fig.update_yaxes(title_text="A", row=1, col=1)
    
    return fig

# --- RENDER TABS ---
if not dff.empty:
    tab1, tab2 = st.tabs(["📈 Combined View", "🔲 Grid View"])

    with tab1:
        st.plotly_chart(render_plotly_combined(dff), use_container_width=True)

    with tab2:
        st.plotly_chart(render_plotly_grid(dff), use_container_width=True)
else:
    st.info("Pilih filter di samping untuk menampilkan data.")

# Footer
st.markdown("---")
st.caption(f"Powered by Operation & Maintenance Team - Cahaya Power Indonesia | {site_selected}")