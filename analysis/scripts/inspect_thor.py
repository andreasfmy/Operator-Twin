import pandas as pd, glob, os

folder = "outputs/data/thor_magni/scenario_3/Scenario_3"
files = sorted(glob.glob(os.path.join(folder, "*.csv")))
print("FILES:", len(files))
print("READING:", os.path.basename(files[0]))

df = pd.read_csv(files[0])

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- HEAD ---")
print(df.head(10).to_string())

for col in ["agent_type", "ag_id"]:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False))

print("\n--- NUMERIC RANGES ---")
print(df.describe().to_string())