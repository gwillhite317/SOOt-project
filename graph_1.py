import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


df = pd.read_csv("C:/Users/mowma/Downloads/soot_STAQS_20260212_582d0e3e665f312e6e57a44eb6bc85d9/soot_STAQS_20260212_582d0e3e665f312e6e57a44eb6bc85d9_CSV.csv")
# Force ozone to numeric
df["Ozone_ppbv"] = pd.to_numeric(df["Ozone_ppbv"], errors="coerce")
df["Datetime_Mid"] = pd.to_datetime(df["Datetime_Mid"], errors="coerce")
df.loc[df["Ozone_ppbv"] < 0, "Ozone_ppbv"] = np.nan
df = df.dropna(subset=["Datetime_Mid", "Ozone_ppbv"]).sort_values("Datetime_Mid")
df = df.dropna(subset=["Datetime_Mid"]).sort_values("Datetime_Mid")
# Index by time
ts = df.set_index("Datetime_Mid")["Ozone_ppbv"]

# Finer time grid -> smoother visual (try "10s", "5s", "1s" depending on density)
grid = "10s"
ts_reg = ts.resample(grid).mean().interpolate("time")

# Gentle smoothing (choose ONE)
smooth = ts_reg.rolling(window=21, center=True, min_periods=1).mean()  # ~21*10s = 3.5 min smoothing

plt.figure(figsize=(11, 5.8), dpi=150)
plt.plot(ts_reg.index, ts_reg.values, linewidth=1.0, alpha=0.35, label=f"Ozone ({grid} grid)")
plt.plot(smooth.index, smooth.values, linewidth=2.2, label="Smoothed")

plt.title("NASA SOOT STAQS — Ozone (smoothed)")
plt.xlabel("Time (Datetime_Mid)")
plt.ylabel("Ozone (ppbv)")
plt.grid(True, alpha=0.25)

loc = mdates.AutoDateLocator(minticks=6, maxticks=10)
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
