import os
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="outputs/data/thor_magni/scenario_3/Scenario_3")
ap.add_argument("--outdir", default="outputs")
ap.add_argument("--n-perm", type=int, default=100,
                help="numero de sorteos de permutacion a promediar")
ap.add_argument("--seed-perm", type=int, default=7)
ap.add_argument("--seed-boot", type=int, default=11)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--min-frames", type=int, default=20)
ap.add_argument("--min-perm-frac", type=float, default=0.5,
                help="fraccion minima de sorteos en que una unidad debe aparecer")
ap.add_argument("--threshold", type=float, default=90.0)
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
RNG_PERM = np.random.default_rng(args.seed_perm)
RNG_BOOT = np.random.default_rng(args.seed_boot)

BINS = [0, 1.5, 2.5, 3.5, 5.0, 8.0, 100.0]
LABELS = ["<1.5", "1.5-2.5", "2.5-3.5", "3.5-5", "5-8", ">8"]
FIT_BANDS = ["<1.5", "1.5-2.5", "2.5-3.5", "3.5-5"]
KEY = ["file", "ag_id", "band"]


def load(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No se encontraron CSV en {folder!r}. "
                         "Revisa --folder.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["file"] = os.path.basename(f)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw[["x", "y"]] = raw[["x", "y"]] * 1e-3
    print(f"archivos leidos: {len(files)}")
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


def per_unit(m, thr, min_frames):
    """Una fila por persona-corrida-banda. Evita inflar n por autocorrelacion."""
    g = (m.groupby(KEY, observed=False)
           .agg(frac=("rel", lambda s: (s < thr).mean()),
                d_mean=("dist", "mean"),
                n=("rel", "size"))
           .reset_index())
    return g[g["n"] >= min_frames]


# ------------------------------------------------------------------ carga
raw = load(args.folder)
humans = raw[raw["ag_id"].str.startswith("Helmet")].copy()
robot = (raw[raw["ag_id"] == "DARKO_Robot"][["file", "frame_id", "x", "y"]]
         .rename(columns={"x": "rx", "y": "ry"}))

real = geometry(humans, robot)
u_real = per_unit(real, args.threshold, args.min_frames)
print(f"filas real={len(real)}   unidades persona-corrida-banda={len(u_real)}")

# ------------------------------------------------- nulo promediado sobre N
print(f"\npromediando el nulo sobre {args.n_perm} sorteos de permutacion...")
null_chunks = []
for i in range(args.n_perm):
    nm = geometry(humans, permute(robot, RNG_PERM))
    un = per_unit(nm, args.threshold, args.min_frames)
    un = un[KEY + ["frac"]].copy()
    un["draw"] = i
    null_chunks.append(un)
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  sorteo {i + 1}/{args.n_perm}")

null_all = pd.concat(null_chunks, ignore_index=True)

# dispersion del nulo entre sorteos: esto es lo que un solo sorteo ocultaba
spread = (null_all.groupby(["band", "draw"], observed=False)["frac"].mean()
                  .groupby("band", observed=False)
                  .agg(nulo_medio="mean", nulo_sd="std",
                       nulo_min="min", nulo_max="max"))
print("\n=== DISPERSION DEL NULO ENTRE SORTEOS ===")
print("(un solo sorteo te daba un punto cualquiera de este rango)")
print(spread.reindex(LABELS).round(4).to_string())

u_null = (null_all.groupby(KEY, observed=False)
                  .agg(frac=("frac", "mean"), draws=("frac", "size"))
                  .reset_index())
u_null = u_null[u_null["draws"] >= args.min_perm_frac * args.n_perm]

# ------------------------------------------------------- bootstrap pareado
pair = u_real.merge(u_null, on=KEY, suffixes=("_r", "_n"))
pair = pair.dropna(subset=["frac_r", "frac_n"])

print(f"\numbral +/-{args.threshold:.0f} grados, bootstrap pareado, "
      f"{args.n_boot} replicas, nulo = media de {args.n_perm} sorteos")
print(f"{'banda':>10} {'pares':>6} {'d_med':>6} {'real':>7} {'nulo':>7} "
      f"{'exceso':>8} {'IC 95%':>18}")

rows = []
unit_excess = {}
for b in LABELS:
    sub = pair[pair["band"] == b]
    if len(sub) < 5:
        continue
    d = (sub["frac_r"] - sub["frac_n"]).to_numpy()
    boot = RNG_BOOT.choice(d, (args.n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    flag = "  *" if lo > 0 or hi < 0 else ""
    dm = sub["d_mean"].mean()
    print(f"{b:>10} {len(sub):>6} {dm:>6.2f} {sub['frac_r'].mean():>7.3f} "
          f"{sub['frac_n'].mean():>7.3f} {d.mean():>8.3f} "
          f"[{lo:>6.3f},{hi:>6.3f}]{flag}")
    rows.append({"band": b, "d": dm, "real": sub["frac_r"].mean(),
                 "null": sub["frac_n"].mean(), "excess": d.mean(),
                 "lo": lo, "hi": hi, "n": len(sub)})
    unit_excess[b] = d

prof = pd.DataFrame(rows)
prof.to_csv(os.path.join(args.outdir, "alpha_profile_multiperm.csv"), index=False)

# ------------------------------------------------------------- ajuste
model = lambda d, A, tau: A * np.exp(-d / tau)
floor = prof["null"].mean()
fit = prof[prof["band"].isin(FIT_BANDS)]
if len(fit) < 3:
    raise SystemExit("muy pocas bandas para ajustar")

(A, tau), cov = curve_fit(model, fit["d"], fit["excess"], p0=[0.2, 2.0],
                          maxfev=20000)
se = np.sqrt(np.diag(cov))

# IC del ajuste por bootstrap sobre person-runs (no sobre las 4 bandas)
fit_boot = []
for _ in range(args.n_boot // 4):
    ex = []
    for b in FIT_BANDS:
        d = unit_excess[b]
        ex.append(RNG_BOOT.choice(d, len(d), replace=True).mean())
    try:
        p, _ = curve_fit(model, fit["d"], ex, p0=[0.2, 2.0], maxfev=20000)
        if 0 < p[1] < 50:
            fit_boot.append(p)
    except RuntimeError:
        pass
fit_boot = np.array(fit_boot)

print(f"\npiso (media del nulo)      = {floor:.4f}")
print(f"A   (exceso a d=0)         = {A:.4f}  +/- {se[0]:.4f} (curve_fit)")
print(f"tau (decaimiento, m)       = {tau:.3f}  +/- {se[1]:.3f} (curve_fit)")
if len(fit_boot) > 50:
    A_lo, A_hi = np.percentile(fit_boot[:, 0], [2.5, 97.5])
    t_lo, t_hi = np.percentile(fit_boot[:, 1], [2.5, 97.5])
    print(f"\nIC 95% por bootstrap sobre person-runs ({len(fit_boot)} ajustes validos):")
    print(f"A   : [{A_lo:.3f}, {A_hi:.3f}]")
    print(f"tau : [{t_lo:.3f}, {t_hi:.3f}] m")
print(f"\nbandas usadas              = {', '.join(FIT_BANDS)}")
print(f"\n  alpha_dist(d) = {floor:.3f} + {A:.3f} * exp(-d / {tau:.2f})")

pred = floor + model(fit["d"].to_numpy(), A, tau)
print(f"\n{'banda':>10} {'d':>6} {'obs':>7} {'ajuste':>7} {'resid':>7}")
for (_, r), p in zip(fit.iterrows(), pred):
    print(f"{r['band']:>10} {r['d']:>6.2f} {r['real']:>7.3f} {p:>7.3f} "
          f"{r['real'] - p:>+7.3f}")

print(f"\nperfil guardado en {args.outdir}/alpha_profile_multiperm.csv")