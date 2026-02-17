# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Ozone vs Altitude", layout="wide")

# --- Config ---
DEFAULT_PATH = r"C:/Users/mowma/Downloads/soot_STAQS_20260212_582d0e3e665f312e6e57a44eb6bc85d9/soot_STAQS_20260212_582d0e3e665f312e6e57a44eb6bc85d9_CSV.csv"
FILL_VALUES = [-9999, -9999.0, -8888, -8888.0, -7777, -7777.0]

ALT_COL = "Altitude_m_MSL"
O3_COL  = "Ozone_ppbv"

st.title("NASA SOOT STAQS — Ozone vs Altitude")

# --- Sidebar controls ---
st.sidebar.header("Controls")
path = st.sidebar.text_input("CSV path", value=DEFAULT_PATH)
bin_m = st.sidebar.slider("Altitude bin size (m)", min_value=10, max_value=500, value=50, step=10)
window = st.sidebar.slider("Rolling window (bins)", min_value=3, max_value=51, value=11, step=2)
show_ci = st.sidebar.checkbox("Show ~95% CI band (SEM)", value=True)
show_raw = st.sidebar.checkbox("Show raw scatter", value=True)

@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, na_values=FILL_VALUES)

try:
    df = load_data(path)
except Exception as e:
    st.error(f"Could not read CSV. Error: {e}")
    st.stop()

# --- Force numeric & clean ---
df[ALT_COL] = pd.to_numeric(df.get(ALT_COL), errors="coerce")
df[O3_COL]  = pd.to_numeric(df.get(O3_COL),  errors="coerce")

df.loc[df[O3_COL] <= 0, O3_COL] = np.nan  # negative ozone not physical
d = df.dropna(subset=[ALT_COL, O3_COL]).copy()

if d.empty:
    st.warning("No valid rows after cleaning. Check column names and fill values.")
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

# --- Plot styling ---
# ----- Plot (vertical profile: ozone on x, altitude on y) -----
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8, 7))

# Color palette (feel free to tweak)
c_raw    = "#969CA1"   # grey
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

# Optional: make altitude start at 0 if you want
# ax.set_ylim(bottom=0)

# Optional: if you prefer altitude increasing downward (typical sounding plots sometimes invert)
# ax.invert_yaxis()

ax.legend(frameon=False, loc="best")
fig.tight_layout()

# Streamlit render (use this instead of plt.show())
st.pyplot(fig, clear_figure=False)
