import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

# === Step 1. Load IMF WEO CSV ===
weo = pd.read_csv("WEO_Data.csv")

# Keep only Pakistan
pak = weo[weo["Country"] == "Pakistan"]

# Identify year columns (2010–2030 in WEO file)
year_cols = [str(y) for y in range(2010, 2031)]
pak[year_cols] = pak[year_cols].apply(pd.to_numeric, errors="coerce")

# === Step 2. Filter GDP growth & Inflation ===
gdp = pak[
    (pak["Subject Descriptor"] == "Gross domestic product, constant prices") &
    (pak["Units"] == "Percent change")
]

infl = pak[
    (pak["Subject Descriptor"] == "Inflation, average consumer prices") &
    (pak["Units"] == "Percent change")
]

# Extract as time series
gdp_growth = gdp[year_cols].T.rename(columns={gdp.index[0]: "GDP_Growth"})
inflation = infl[year_cols].T.rename(columns={infl.index[0]: "Inflation"})

# Combine
macro = pd.concat([gdp_growth, inflation], axis=1)
macro.index = macro.index.astype(int)

# === Step 3. Fit VAR model ===
model = VAR(macro.dropna())
results = model.fit(maxlags=2)

# === Step 4. Forecast baseline ===
forecast_steps = 5
lag_order = results.k_ar
input_data = macro.values[-lag_order:]
forecast = results.forecast(y=input_data, steps=forecast_steps)

forecast_df = pd.DataFrame(
    forecast,
    index=range(macro.index[-1] + 1, macro.index[-1] + 1 + forecast_steps),
    columns=["GDP_Growth_Forecast", "Inflation_Forecast"]
)

# === Step 5. Scenario definitions (quantiles) ===
Best = forecast_df.quantile(0.8)
base = forecast_df.quantile(0.5)
Worst = forecast_df.quantile(0.2)

scenarios = pd.DataFrame({
    "Best": Best,
    "Base": base,
    "Worst": Worst
})

# === Step 6. Monte Carlo simulation for weights ===
n_sim = 1000
sim_results = []

for i in range(n_sim):
    sim_path = results.simulate_var(steps=forecast_steps)
    sim_results.append(sim_path[-1])  # take last simulated year

sim_results = np.array(sim_results)
sim_df = pd.DataFrame(sim_results, columns=["GDP_Growth", "Inflation"])

# Classify scenarios based on quantile thresholds
gdp_q20, gdp_q80 = sim_df["GDP_Growth"].quantile([0.2, 0.8])
infl_q20, infl_q80 = sim_df["Inflation"].quantile([0.2, 0.8])

def classify(row):
    if (row["GDP_Growth"] >= gdp_q80) and (row["Inflation"] <= infl_q20):
        return "Best"
    elif (row["GDP_Growth"] <= gdp_q20) and (row["Inflation"] >= infl_q80):
        return "Worst"
    else:
        return "Base"

sim_df["Scenario"] = sim_df.apply(classify, axis=1)

# Compute weights = frequency of each scenario
scenario_weights = sim_df["Scenario"].value_counts(normalize=True).to_dict()

print("\n=== Forecast ===")
print(forecast_df)

print("\n=== Scenario Values ===")
print(scenarios)

print("\n=== Scenario Weights (from simulation) ===")
print(scenario_weights)

# Save outputs
forecast_df.to_csv("macro_forecast.csv")
scenarios.to_csv("macro_scenarios.csv")
sim_df.to_csv("macro_simulations.csv")
scenario_weights = sim_df["Scenario"].value_counts(normalize=True)
scenario_weights.to_csv("scenario_weights.csv")
