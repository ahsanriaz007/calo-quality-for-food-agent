import numpy as np
import pandas as pd


def generate_kitchen_logs(num_records=100):
  """Generates synthetic kitchen operational logs for testing."""
  np.random.seed(42)

  kitchens = [
      "Riyadh-Central",
      "Jeddah-North",
      "Dubai-Marina",
      "Manama-Main",
      "Kuwait-City",
  ]

  data = []
  for i in range(num_records):
    # Standard cold storage temperature (Target: 2-6°C)
    fridge_temp = round(np.random.normal(loc=4.0, scale=2.5), 1)

    # Hot meal cook temperature (Target: >= 74°C for safety)
    cook_temp = round(np.random.normal(loc=76.0, scale=5.0), 1)

    # Prep time in minutes (Target: < 25 mins)
    prep_time = int(np.random.normal(loc=18, scale=6))

    data.append({
        "log_id": f"LOG-{1000 + i}",
        "kitchen": np.random.choice(kitchens),
        "fridge_temp_c": fridge_temp,
        "cook_temp_c": cook_temp,
        "prep_time_mins": max(5, prep_time),
    })

  return pd.DataFrame(data)


def detect_anomalies(df):
  """Flags operational rule violations based on food safety standards."""
  flagged = []

  for idx, row in df.iterrows():
    violations = []

    # HACCP Cold Chain Check (Above 6°C is a spoilage risk)
    if row["fridge_temp_c"] > 6.0:
      violations.append(
          f"High Fridge Temp ({row['fridge_temp_c']}°C > 6.0°C)"
      )
    elif row["fridge_temp_c"] < 0.0:
      violations.append(f"Freezing Risk ({row['fridge_temp_c']}°C < 0.0°C)")

    # Food Safety Internal Cook Temp Check (< 74°C risk of undercooking)
    if row["cook_temp_c"] < 74.0:
      violations.append(f"Undercooked Item ({row['cook_temp_c']}°C < 74.0°C)")

    # Prep Delay (> 30 mins)
    if row["prep_time_mins"] > 30:
      violations.append(f"Prep Delay ({row['prep_time_mins']} mins > 30 mins)")

    if violations:
      flagged.append({
          "log_id": row["log_id"],
          "kitchen": row["kitchen"],
          "issue_summary": " | ".join(violations),
          "severity": "HIGH" if len(violations) > 1 else "MEDIUM",
      })

  return pd.DataFrame(flagged)


if __name__ == "__main__":
  print("Generating kitchen operational logs...")
  df_logs = generate_kitchen_logs(150)

  print("Running HACCP anomaly detector...")
  anomalies = detect_anomalies(df_logs)

  # Save flagged events for our dashboard
  anomalies.to_csv("kitchen_anomalies.csv", index=False)

  print(
      f"Analysis Complete: Flagged {len(anomalies)} operational anomalies out"
      " of 150 events."
  )
  print("\nSample Flagged Events:")
  print(anomalies.head())