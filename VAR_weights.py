import pandas as pd
import numpy as np
import argparse
from statsmodels.tsa.api import VAR


def cal_senario_weights(event):


    pak = pd.read_csv(event["WEO_Data"])


    year_cols = [str(y) for y in range(2010, 2025)] #2024+1
    pak[year_cols] = pak[year_cols].apply(pd.to_numeric, errors="coerce")


    gdp = pak[
        (pak["Subject Descriptor"] == "Gross domestic product, constant prices") &
        (pak["Units"] == "Percent change")
    ]

    infl = pak[
        (pak["Subject Descriptor"] == "Inflation, average consumer prices") &
        (pak["Units"] == "Percent change")
    ]

    gdp_growth = gdp[year_cols].T.rename(columns={gdp.index[0]: "GDP_Growth"})
    inflation = infl[year_cols].T.rename(columns={infl.index[0]: "Inflation"})

    macro = pd.concat([gdp_growth, inflation], axis=1)

    macro.index = macro.index.astype(int)

    model = VAR(macro.dropna())
    results = model.fit(maxlags=2)

    forecast_steps = 5
    lag_order = results.k_ar
    input_data = macro.values[-lag_order:]


    forecast = results.forecast(y=input_data, steps=forecast_steps)

    forecast_df = pd.DataFrame(
        forecast,
        index=range(macro.index[-1] + 1, macro.index[-1] + 1 + forecast_steps),
        columns=["GDP_Growth_Forecast", "Inflation_Forecast"]
    )


    GDP_Growth_Forecast = forecast_df['GDP_Growth_Forecast']
    GDP_Growth_Forecast_optimistic = GDP_Growth_Forecast.quantile(0.8)
    GDP_Growth_Forecast_base = GDP_Growth_Forecast.quantile(0.5)
    GDP_Growth_Forecast_pessimistic = GDP_Growth_Forecast.quantile(0.2)

    Inflation_Forecast = forecast_df['Inflation_Forecast']
    Inflation_Forecast_optimistic = Inflation_Forecast.quantile(0.2)
    Inflation_Forecast_base = Inflation_Forecast.quantile(0.5)
    Inflation_Forecast_pessimistic = Inflation_Forecast.quantile(0.8)


    index = ['GDP_Growth_Forecast', 'Inflation_Forecast']

    scenarios = {
        'Optimistic': [GDP_Growth_Forecast_optimistic,Inflation_Forecast_optimistic],
        'Base': [GDP_Growth_Forecast_base,Inflation_Forecast_base],
        'Pessimistic': [GDP_Growth_Forecast_pessimistic,Inflation_Forecast_pessimistic,]
    }

    scenarios = pd.DataFrame(scenarios,index=index)

    n_sim = 50000
    sim_results = []

    for i in range(n_sim):
        sim_path = results.simulate_var(steps=forecast_steps)
        sim_results.append(sim_path[-1])  # take last simulated year

    sim_results = np.array(sim_results)
    sim_df = pd.DataFrame(sim_results, columns=["GDP_Growth", "Inflation"])

    gdp_q20, gdp_q80 = sim_df["GDP_Growth"].quantile([0.2, 0.8])
    infl_q20, infl_q80 = sim_df["Inflation"].quantile([0.2, 0.8])

    def classify(row):
        if (row["GDP_Growth"] >= gdp_q80) and (row["Inflation"] <= infl_q20):
            return "Optimistic"
        elif (row["GDP_Growth"] <= gdp_q20) and (row["Inflation"] >= infl_q80):
            return "Pessimistic"
        else:
            return "Base"

    sim_df["Scenario"] = sim_df.apply(classify, axis=1)

    scenario_weights = sim_df["Scenario"].value_counts(normalize=True).to_dict()

    print("\n=== Forecast ===")
    print(forecast_df)

    print("\n=== Scenario Values ===")
    print(scenarios)

    print("\n=== Scenario Weights (from simulation) ===")
    print(scenario_weights)
    


    print("\n=== Results Summary ===")
    print(results.summary())

    if event["save_csv"] == "yes":
        forecast_df.to_csv(r"results\macro_forecast.csv")
        scenarios.to_csv(r"results\macro_scenarios.csv")
        sim_df.to_csv(r"results\macro_simulations.csv")
        scenario_weights = sim_df["Scenario"].value_counts(normalize=True)
        scenario_weights.to_csv(r"results\scenario_weights.csv")

    return sim_df,scenario_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--WEO_Data', type=str, default=r'inputs\WEO_Data.csv')
    parser.add_argument('--save_csv', type=str, default='yes')
    args = parser.parse_args()
    event = vars(args)
    cal_senario_weights(event)
