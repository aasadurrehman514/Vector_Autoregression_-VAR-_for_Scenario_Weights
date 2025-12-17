# Vector_Autoregression_-VAR-_for_Scenario_Weights
Vector Autoregression (VAR) for Scenario Weights
# VAR-Based Scenario Weights for CECL / IFRS 9

This project implements a **Vector Autoregression (VAR)–based framework** to derive macroeconomic scenario weights for Expected Credit Loss (ECL) estimation, inspired by Moody’s Analytics (2019).

## Overview
Accounting standards such as CECL and IFRS 9 require lifetime ECL estimates that incorporate **forward-looking macroeconomic information**. While guidance emphasizes the use of multiple scenarios, it does not prescribe how scenario weights should be determined.

This framework:
- Models joint macro dynamics using VAR
- Simulates thousands of future macroeconomic paths
- Derives scenario weights from the empirical distribution of outcomes

## Methodology
1. **Input Data**
   - Historical GDP growth and inflation (IMF WEO data)

2. **Macro Model**
   - Fit a VAR model to capture joint dynamics

3. **Simulation**
   - Generate large-scale Monte Carlo simulations of future macro paths

4. **Scenario Classification**
   - Optimistic: high GDP growth, low inflation  
   - Base: central outcomes  
   - Pessimistic: low GDP growth, high inflation

5. **Scenario Weights**
   - Calculated as the proportion of simulated outcomes in each scenario

## Key Insight
Scenario weights are treated as **approximations of probability mass**, not subjective recession probabilities—consistent with CECL principles.

## Outputs
- Forecasted macro variables
- Scenario centroids
- Simulated macro distributions
- Scenario weights
- Animated visualizations of simulation convergence

## References
Moody’s Analytics (2019): *Deconstructing Scenario Weights for CECL* 

