# strem_2.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Ozone vs Altitude", layout="wide")

# --- Config ---
# CSV is committed to the repo (main branch) at the repo root
DATA_PATH = Path(__file__).parent / "soot_trimmed.csv"

FILL_VALUES = [-9999, -9999.0, -8888, -8888.0, -7777, -7777.0]

ALT_COL = "Altitude_m_MSL"
O3_COL = "Ozone_ppbv"

st.title("NASA SOOT STAQS — Ozone vs Altitude")

# --- Sidebar controls ---
st.sidebar.header("Controls")
bin_m = st.sidebar.slider("Altitude bin size (m)", min_value=10, max_value=500, value=50, step=10)
window = st.sidebar.slider("Rolling window (bins)", min_value=3, max_value=51, value=11, step=2)
show_ci = st.sidebar.checkbox("Show ~95% CI band (SEM)", value=True)
show_raw = st.sidebar.checkbox("Show raw scatter", value=True)

@st.cache_data(show_spinner=False)
def load_data(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path, na_values=FILL_VALUES)

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not read CSV at {DATA_PATH}. Error: {e}")
    st.stop()

# --- Validate columns early ---
missing = [c for c in [ALT_COL, O3_COL] if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {missing}. Found columns: {list(df.columns)}")
    st.stop()

# --- Force numeric & clean ---
df[ALT_COL] = pd.to_numeric(df[ALT_COL], errors="coerce")
df[O3_COL] = pd.to_numeric(df[O3_COL], errors="coerce")

# negative or zero ozone not physical (and may include fills that slipped through)
df.loc[df[O3_COL] <= 0, O3_COL] = np.nan

d = df.dropna(subset=[ALT_COL, O3_COL]).copy()

if d.empty:
    st.warning("No valid rows after cleaning. Check fill values, column names, and data.")
    st.stop()

# --- Smooth profile via altitude binning ---
d["alt_bin"] = (d[ALT_COL] / bin_m).round() * bin_m

profile = (
    d.groupby("alt_bin")[O3_COL]
      .agg(mean="mean", median="median", n="size", std="std")
      .reset_index()
      .sort_values("alt_bin")
)

profile["sem"] = profile["std"] / np.sqrt(profile["n"])
profile.loc[profile["n"] < 5, ["sem"]] = np.nan

profile["mean_smooth"] = (
    profile["mean"]
      .rolling(window=window, center=True, min_periods=3)
      .mean()
)

# --- Plot styling ---# ----- Plot (vertical profile: ozone on x, altitude on y) -----
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8, 7))

# Color palette (feel free to tweak)
c_raw    = "#9aa0a6"   # grey
c_binned = "#1f77b4"   # blue
c_smooth = "#d62728"   # red
c_ci     = "#ff7f0e"   # orange

# 1) Raw scatter (ozone vs altitude)
ax.scatter(
    d[O3_COL], d[ALT_COL],
    s=7, alpha=0.10, linewidths=0,
    color=c_raw, label="Raw"
)

# 2) Binned mean (x=mean ozone, y=alt_bin)
ax.plot(
    profile["mean"], profile["alt_bin"],
    linewidth=1.4, alpha=0.70,
    color=c_binned, label=f"Binned mean ({bin_m} m)"
)

# 3) Smoothed profile (x=mean_smooth ozone, y=alt_bin)
ax.plot(
    profile["mean_smooth"], profile["alt_bin"],
    linewidth=2.6,
    color=c_smooth, label=f"Smoothed (rolling {window} bins)"
)

# 4) CI band around smoothed line: fill between x-lower and x-upper along y
mask = profile["sem"].notna() & profile["mean_smooth"].notna()
if mask.any():
    lower = profile.loc[mask, "mean_smooth"] - 1.96 * profile.loc[mask, "sem"]
    upper = profile.loc[mask, "mean_smooth"] + 1.96 * profile.loc[mask, "sem"]

    ax.fill_betweenx(
        y=profile.loc[mask, "alt_bin"],
        x1=lower,
        x2=upper,
        alpha=0.18,
        color=c_ci,
        label="~95% CI (SEM)"
    )

# Labels/title
ax.set_title("NASA SOOT STAQS — Vertical Ozone Profile (cleaned)")
ax.set_xlabel("Ozone (ppbv)")
ax.set_ylabel("Altitude (m MSL)")

# Nice grid
ax.grid(True, alpha=0.22)



ax.legend(frameon=False, loc="best")
fig.tight_layout()

# Streamlit render (use this instead of plt.show())
st.pyplot(fig, clear_figure=False)
