import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = "outputs/data/thor_magni/scenario_3/Scenario_3"
OUTDIR = "outputs/figures"
os.makedirs(OUTDIR, exist_ok=True)

frames = []
for f in sorted(glob.glob(os.path.join(FOLDER, "*.csv"))):
    df = pd.read_csv(f)
    df["file"] = os.path.basename(f)
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)

for c in ["x", "y"]:
    raw[c] = raw[c] * 1e-3

raw = raw.dropna(subset=["x", "y", "rot_yaw"])
raw = raw[raw["ag_id"].str.startswith("Helmet")]
raw = raw.sort_values(["file", "ag_id", "Time"])

g = raw.groupby(["file", "ag_id"])
raw["dt"] = g["Time"].diff()
raw["dx"] = g["x"].diff()
raw["dy"] = g["y"].diff()
raw["speed"] = np.sqrt(raw["dx"]**2 + raw["dy"]**2) / raw["dt"]

# only frames where the person is clearly walking
w = raw[(raw["speed"] > 0.6) & (raw["speed"] < 2.5) & (raw["dt"] < 0.5)].copy()

w["heading"] = np.degrees(np.arctan2(w["dy"], w["dx"]))
diff = (w["rot_yaw"] - w["heading"] + 180) % 360 - 180
w["yaw_minus_heading"] = diff

print("samples used:", len(w))
print("\n=== yaw - heading (deg) ===")
print(w["yaw_minus_heading"].describe(percentiles=[.25,.5,.75]))

# circular mean = the real offset
ang = np.radians(w["yaw_minus_heading"])
offset = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
R = np.sqrt(np.sin(ang).mean()**2 + np.cos(ang).mean()**2)
print(f"\ncircular mean offset : {offset:.1f} deg")
print(f"concentration R      : {R:.3f}   (0 = uniform, 1 = perfectly aligned)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(w["yaw_minus_heading"], bins=72, range=(-180, 180))
ax.axvline(offset, color="r", ls="--", label=f"circular mean = {offset:.1f} deg")
ax.set_xlabel("rot_yaw - walking heading (deg)")
ax.set_ylabel("samples")
ax.set_title("Helmet yaw convention check")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUTDIR}/04_yaw_calibration.png", dpi=150)
print(f"\nFigure written to {OUTDIR}/04_yaw_calibration.png")