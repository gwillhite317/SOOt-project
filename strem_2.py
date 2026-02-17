# strem_2.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # important for cloud/headless
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Ozone vs Altitude", layout="wide")

# --- Config ---
DATA_PATH = Path(__file__).parent / "soot_trimmed.csv"
FILL_VALUES = [-9999, -9999.0, -8888, -8888.0, -7777, -7777.0]

ALT_COL = "Altitude_m_MSL"
O3_COL = "Ozone_ppbv"

st.title("NASA SOOT STAQS — Ozone vs Altitude")

# --- Sidebar controls ---
st.sidebar.write("Running:", __file__)
st.sidebar.write("Commit marker: v1")
st.sidebar.header("Controls")
bin_m = st.sidebar.slider("Altitude bin size (m)", 10, 500, 50, 10, key="bin_m")
window = st.sidebar.slider("Rolling window (bins)", 3, 51, 11, 2, key="window")
show_raw = st.sidebar.checkbox("Show raw scatter", value=True, key="show_raw")
show_ci  = st.sidebar.checkbox("Show ~95% CI band (SEM)", value=True, key="show_ci")

@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, na_values=FILL_VALUES)

def prep_profile(df: pd.DataFrame, bin_m: int, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Validate columns
    missing = [c for c in [ALT_COL, O3_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found columns: {list(df.columns)}")

    # Force numeric & clean
    x = df.copy()
    x[ALT_COL] = pd.to_numeric(x[ALT_COL], errors="coerce")
    x[O3_COL]  = pd.to_numeric(x[O3_COL], errors="coerce")
    x.loc[x[O3_COL] <= 0, O3_COL] = np.nan
    d = x.dropna(subset=[ALT_COL, O3_COL]).copy()

    if d.empty:
        raise ValueError("No valid rows after cleaning.")

    # Bin altitude, compute profile stats
    d["alt_bin"] = (d[ALT_COL] / bin_m).round() * bin_m

    profile = (
        d.groupby("alt_bin")[O3_COL]
          .agg(mean="mean", median="median", n="size", std="std")
          .reset_index()
          .sort_values("alt_bin")
    )

    profile["sem"] = profile["std"] / np.sqrt(profile["n"])
    profile.loc[profile["n"] < 5, "sem"] = np.nan

    profile["mean_smooth"] = (
        profile["mean"]
        .rolling(window=window, center=True, min_periods=3)
        .mean()
    )

    return d, profile

def make_profile_plot(
    d: pd.DataFrame,
    profile: pd.DataFrame,
    bin_m: int,
    window: int,
    show_raw: bool,
    show_ci: bool,
) -> matplotlib.figure.Figure:

    # Build a fresh figure every rerun (no global pyplot state)
    fig = matplotlib.figure.Figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111)

    # Minimal style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colors
    c_raw    = "#9aa0a6"
    c_binned = "#1f77b4"
    c_smooth = "#d62728"
    c_ci     = "#ff7f0e"

    # Raw scatter (x=ozone, y=altitude)
    if show_raw:
        ax.scatter(
            d[O3_COL], d[ALT_COL],
            s=7, alpha=0.10, linewidths=0,
            color=c_raw, label="Raw"
        )

    # Binned mean (x=mean ozone, y=alt_bin)
    ax.plot(
        profile["mean"], profile["alt_bin"],
        linewidth=1.4, alpha=0.75,
        color=c_binned, label=f"Binned mean ({bin_m} m)"
    )

    # Smoothed mean
    ax.plot(
        profile["mean_smooth"], profile["alt_bin"],
        linewidth=2.6,
        color=c_smooth, label=f"Smoothed (rolling {window} bins)"
    )

    # CI band around smoothed line
    if show_ci:
        mask = profile["sem"].notna() & profile["mean_smooth"].notna()
        if mask.any():
            lower = profile.loc[mask, "mean_smooth"] - 1.96 * profile.loc[mask, "sem"]
            upper = profile.loc[mask, "mean_smooth"] + 1.96 * profile.loc[mask, "sem"]

            ax.fill_betweenx(
                y=profile.loc[mask, "alt_bin"],
                x1=lower, x2=upper,
                alpha=0.18, color=c_ci,
                label="~95% CI (SEM)"
            )

    ax.set_title("Vertical Ozone Profile (cleaned)")
    ax.set_xlabel("Ozone (ppbv)")
    ax.set_ylabel("Altitude (m MSL)")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    return fig

# ---- Load data ----
try:
    df = load_data(str(DATA_PATH))
except Exception as e:
    st.error(f"Could not read CSV at {DATA_PATH}. Error: {e}")
    st.stop()

# ---- Prep + plot ----
try:
    d, profile = prep_profile(df, bin_m=bin_m, window=window)
except Exception as e:
    st.error(str(e))
    st.stop()

fig = make_profile_plot(
    d, profile,
    bin_m=bin_m,
    window=window,
    show_raw=show_raw,
    show_ci=show_ci
)

# Render
st.pyplot(fig)

