import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = "C:/Users/mowma/Downloads/soot_STAQS_20260212_582d0e3e665f312e6e57a44eb6bc85d9/soot_STAQS_20260212_582d0e3e665f312e6e57a44eb6bc85d9_CSV.csv"

FILL_VALUES = [-9999, -9999.0, -8888, -8888.0, -7777, -7777.0]

df = pd.read_csv(PATH, na_values=FILL_VALUES)

# --- Columns (adjust if yours differ) ---
ALT_COL = "Altitude_m_MSL"
O3_COL  = "Ozone_ppbv"

# Force numeric
df[ALT_COL] = pd.to_numeric(df[ALT_COL], errors="coerce")
df[O3_COL]  = pd.to_numeric(df[O3_COL],  errors="coerce")

# Clean ozone: negative not physical
df.loc[df[O3_COL] <= 0, O3_COL] = np.nan

# Drop rows missing either
d = df.dropna(subset=[ALT_COL, O3_COL]).copy()

# Optional: remove insane altitude outliers if present
# d = d[(d[ALT_COL] > -500) & (d[ALT_COL] < 25000)]

# ----- Smooth profile via altitude binning -----
bin_m = 50  # altitude bin size in meters (try 25, 50, 100)
d["alt_bin"] = (d[ALT_COL] / bin_m).round() * bin_m

profile = (
    d.groupby("alt_bin")[O3_COL]
      .agg(mean="mean", median="median", n="size", std="std")
      .reset_index()
      .sort_values("alt_bin")
)

# Standard error (optional for a light uncertainty ribbon)
profile["sem"] = profile["std"] / np.sqrt(profile["n"])
profile.loc[profile["n"] < 5, ["sem"]] = np.nan  # avoid junky SEM for tiny bins

# Optional additional smoothing of the binned mean (keeps it silky)
# window is in bins, not meters: with 50 m bins, window=11 ~ 550 m smoothing
window = 11
profile["mean_smooth"] = profile["mean"].rolling(window=window, center=True, min_periods=3).mean()

# ----- Plot -----
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(10.5, 6))

# Raw scatter (faint)
ax.scatter(d[ALT_COL], d[O3_COL], s=6, alpha=0.12, linewidths=0, label="Raw")

# Binned mean + smoothed line
ax.plot(profile["alt_bin"], profile["mean"], linewidth=1.2, alpha=0.65, label=f"Binned mean ({bin_m} m)")
ax.plot(profile["alt_bin"], profile["mean_smooth"], linewidth=2.4, label=f"Smoothed profile (rolling {window} bins)")

# Optional SEM band (subtle)
mask = profile["sem"].notna() & profile["mean_smooth"].notna()
ax.fill_between(
    profile.loc[mask, "alt_bin"],
    (profile.loc[mask, "mean_smooth"] - 1.96 * profile.loc[mask, "sem"]),
    (profile.loc[mask, "mean_smooth"] + 1.96 * profile.loc[mask, "sem"]),
    alpha=0.5,
    label="~95% CI (SEM)"
)

ax.set_title("NASA SOOT STAQS — Ozone vs Altitude (cleaned)")
ax.set_xlabel("Altitude (m MSL)")
ax.set_ylabel("Ozone (ppbv)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
