import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = "outputs/data/thor_magni/scenario_3/Scenario_3"
OUTDIR = "outputs/figures"
os.makedirs(OUTDIR, exist_ok=True)
MM_TO_M = 1e-3
MAX_PLAUSIBLE_SPEED = 2.5  # m/s

# ---------------- load ----------------
frames = []
for f in sorted(glob.glob(os.path.join(FOLDER, "*.csv"))):
    name = os.path.basename(f)
    df = pd.read_csv(f)
    df["file"] = name
    df["condition"] = "A-differential" if "SC3A" in name else "B-omni"
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)

for c in ["x", "y", "z"]:
    raw[c] = raw[c] * MM_TO_M

raw = raw.dropna(subset=["x", "y"])
print("rows after dropping NaNs:", len(raw))

# ---------------- speed ----------------
raw = raw.sort_values(["file", "ag_id", "Time"])
g = raw.groupby(["file", "ag_id"])
raw["dt"] = g["Time"].diff()
raw["speed"] = np.sqrt(g["x"].diff()**2 + g["y"].diff()**2) / raw["dt"]
raw.loc[(raw["speed"] > MAX_PLAUSIBLE_SPEED) | (raw["dt"] > 0.5), "speed"] = np.nan

humans = raw[raw["ag_id"].str.startswith("Helmet")].copy()
robot = raw[raw["ag_id"] == "DARKO_Robot"][
    ["file", "frame_id", "x", "y", "rot_yaw"]
].rename(columns={"x": "rx", "y": "ry", "rot_yaw": "r_yaw"})

print("\n=== SPEED BY ROLE (m/s) ===")
print(humans.groupby("agent_type")["speed"].describe()[["count","mean","50%","std"]])

# ---------------- human-robot geometry ----------------
m = humans.merge(robot, on=["file", "frame_id"], how="inner")
m["dist"] = np.sqrt((m["x"] - m["rx"])**2 + (m["y"] - m["ry"])**2)

# angle from human to robot, relative to human head yaw
bearing = np.degrees(np.arctan2(m["ry"] - m["y"], m["rx"] - m["x"]))
rel = (bearing - m["rot_yaw"] + 180) % 360 - 180
m["rel_angle"] = np.abs(rel)

print("\n=== HUMAN-ROBOT DISTANCE (m) ===")
print(m["dist"].describe(percentiles=[.01,.05,.25,.5,.75]))

enc = m.groupby(["file", "ag_id"])["dist"].min().reset_index(name="min_dist")
print("\n=== MIN DISTANCE PER HUMAN PER RUN (m) ===")
print(enc["min_dist"].describe())

close = m[m["dist"] < 3.0]
print("\n=== HEAD ANGLE TO ROBOT WHEN CLOSER THAN 3 m (deg) ===")
print(close["rel_angle"].describe())
print("fraction with robot within +/-60 deg of gaze:",
      round((close["rel_angle"] < 60).mean(), 3))

# ---------------- figures ----------------
fig, ax = plt.subplots(figsize=(8, 5))
for role, sub in humans.groupby("agent_type"):
    s = sub["speed"].dropna()
    if len(s) > 200:
        ax.hist(s, bins=60, range=(0, 2.5), histtype="step", density=True, label=role)
ax.set_xlabel("walking speed (m/s)"); ax.set_ylabel("density")
ax.set_title("Walking speed by role - THOR-MAGNI Scenario 3")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/01_speed_by_role.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(m["dist"], bins=80, range=(0, 12), density=True)
ax.axvline(enc["min_dist"].median(), color="k", ls="--",
           label=f"median min distance = {enc['min_dist'].median():.2f} m")
ax.set_xlabel("human-robot distance (m)"); ax.set_ylabel("density")
ax.set_title("Human-robot separation distribution")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUTDIR}/02_distance.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
h = ax.hist2d(m["dist"], m["rel_angle"], bins=[60, 60],
              range=[[0, 12], [0, 180]], cmap="viridis")
fig.colorbar(h[3], ax=ax, label="samples")
ax.set_xlabel("human-robot distance (m)")
ax.set_ylabel("|angle robot - head yaw| (deg)")
ax.set_title("Head orientation toward robot vs distance")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/03_gaze_vs_distance.png", dpi=150)

print(f"\nFigures written to {OUTDIR}")