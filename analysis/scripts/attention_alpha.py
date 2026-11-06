import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="outputs/data/thor_magni/scenario_3/Scenario_3")
ap.add_argument("--outdir", default="outputs/figures")
ap.add_argument("--seed-perm", type=int, default=7)
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
RNG_PERM = np.random.default_rng(args.seed_perm)
 
BINS = [0, 1.5, 2.5, 3.5, 5.0, 8.0, 100.0]
LABELS = ["<1.5", "1.5-2.5", "2.5-3.5", "3.5-5", "5-8", ">8"]
 
 
def load(folder):
    frames = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = pd.read_csv(f)
        df["file"] = os.path.basename(f)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw[["x", "y"]] = raw[["x", "y"]] * 1e-3
    return raw.dropna(subset=["x", "y", "rot_yaw"])
 
 
def geometry(h, r):
    m = h.merge(r, on=["file", "frame_id"], how="inner")
    m["dist"] = np.hypot(m["x"] - m["rx"], m["y"] - m["ry"])
    b = np.degrees(np.arctan2(m["ry"] - m["y"], m["rx"] - m["x"]))
    m["rel"] = np.abs((b - m["rot_yaw"] + 180) % 360 - 180)
    m["band"] = pd.cut(m["dist"], bins=BINS, labels=LABELS)
    return m
 
 
def permute(robot, rng):
    """Reasigna a cada corrida la trayectoria del robot de otra corrida,
    remapeada sobre los frame_id propios para no perder frames en el join."""
    files = sorted(robot["file"].unique())
    parts = []
    for f in files:
        other = rng.choice([o for o in files if o != f])
        src = robot[robot["file"] == other].sort_values("frame_id")
        tgt_ids = np.sort(robot.loc[robot["file"] == f, "frame_id"].unique())
        idx = np.arange(len(tgt_ids)) % len(src)
        parts.append(pd.DataFrame({
            "file": f, "frame_id": tgt_ids,
            "rx": src["rx"].to_numpy()[idx], "ry": src["ry"].to_numpy()[idx]}))
    return pd.concat(parts, ignore_index=True)
 
 
def profile(m, tag):
    out = m.groupby("band", observed=False).agg(
        n=("rel", "size"),
        dist_mean=("dist", "mean"),
        median_angle=("rel", "median"),
        frac_60=("rel", lambda s: (s < 60).mean()),
        frac_90=("rel", lambda s: (s < 90).mean()))
    print(f"\n=== {tag} ===")
    print(out.round(3).to_string())
    return out
 
 
raw = load(args.folder)
humans = raw[raw["ag_id"].str.startswith("Helmet")].copy()
humans["condition"] = np.where(humans["file"].str.contains("SC3A"),
                               "A-differential", "B-omni")
robot = (raw[raw["ag_id"] == "DARKO_Robot"][["file", "frame_id", "x", "y"]]
         .rename(columns={"x": "rx", "y": "ry"}))
 
real = geometry(humans, robot)
null = geometry(humans, permute(robot, RNG_PERM))
print(f"filas real={len(real)}  null={len(null)}  ratio={len(null)/len(real):.3f}")
 
p_real = profile(real, "REAL")
p_null = profile(null, "PERMUTED NULL")
 
print("\n=== EXCESO (real - null) ===")
exc = pd.DataFrame({"d_mean": p_real["dist_mean"].round(2),
                    "exc_60": (p_real["frac_60"] - p_null["frac_60"]).round(3),
                    "exc_90": (p_real["frac_90"] - p_null["frac_90"]).round(3)})
print(exc.to_string())
 
print("\n=== POR CONDICION, dist < 3 m ===")
close = real[real["dist"] < 3.0]
print(close.groupby("condition")["rel"].describe()[["count", "mean", "50%"]].round(1))
 
centers = p_real["dist_mean"].to_numpy()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(centers, p_real["frac_90"], "o-", color="C0", label="±90° observado")
ax.plot(centers, p_null["frac_90"], "s--", color="C0", alpha=.5, label="±90° nulo")
ax.plot(centers, p_real["frac_60"], "o-", color="C1", label="±60° observado")
ax.plot(centers, p_null["frac_60"], "s--", color="C1", alpha=.5, label="±60° nulo")
ax.set_xlabel("Human-robot distance (m)")
ax.set_ylabel("Time spent within threshold")
ax.set_title("Attention to robot vs distance")
ax.set_xlim(0, 12); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(f"{args.outdir}/05_alpha_vs_distance.png", dpi=150)
print(f"\nFigura: {args.outdir}/05_alpha_vs_distance.png")