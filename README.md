# Passive Portfolio Management — NASDAQ-100 Index Replication

## Overview

This project builds a **simplified 25-asset index fund** that closely replicates the NASDAQ-100 (97 assets) using a **Mixed Integer Programming (MIP)** approach. The model selects assets that maximise pairwise correlation similarity with the full index over four rolling 186-week in-sample windows (Apr 2020 – May 2025).

---

## Problem Formulation

**Objective:** Select q ≤ 25 assets and assign each of the 97 original assets to one selected asset to maximise total correlation-weighted similarity.

| Symbol | Description |
|--------|-------------|
| X[i,j] = 1 | Selected asset i represents original asset j |
| Y[i] = 1 | Asset i is included in the simplified fund |
| ρ[i,j] | Correlation between assets i and j (in-sample window) |

```
maximize  Σᵢ Σⱼ ρ[i,j] · X[i,j]

subject to:
  Σᵢ Y[i] ≤ q                  (fund size ≤ 25)
  X[i,j] ≤ Y[i]   ∀ i,j       (can only assign to selected assets)
  Σᵢ X[i,j] = 1   ∀ j         (each asset assigned to exactly one representative)
  X[i,j], Y[i] ∈ {0,1}
```

**Weight redistribution:** When asset j (index weight wⱼ) is represented by selected asset i, asset i's fund weight accumulates wⱼ.

---

## Data

| File | Description |
|------|-------------|
| `returns_equities_all_instances.xlsx` | Weekly returns, 267 weeks, 97 tickers (Apr 2020 – May 2025) |
| `Correlation_Matrices_all_instances/correlation_matrix{1-4}.xlsx` | Four 97×97 correlation matrices from rolling 186-week windows (offset by 20 weeks) |
| `Sector_Industry_Matrix.xlsx` | 97×23 binary sector/industry membership matrix |
| `index_composition.xlsx` | NASDAQ-100 index weights for all 97 tickers |
| `out_of_sample_validations_all_instances.xlsx` | Original AMPL solver output and return series |

**Rolling windows:**
| Instance | In-Sample Weeks | Dates (approx.) |
|----------|----------------|-----------------|
| 1 | 0–185 | Apr 2020 – Sep 2023 |
| 2 | 20–205 | Sep 2020 – Feb 2024 |
| 3 | 40–225 | Jan 2021 – Jul 2024 |
| 4 | 60–245 | Jun 2021 – Nov 2024 |

---

## File Structure

```
Passive Portfolio Management/
├── README.md
├── solve_optimization.py                # Standalone MIP solver (PuLP/CBC)
├── run_analysis.py                      # Full analysis pipeline (plots + Excel export)
├── 01_data_preprocessing.ipynb          # Data loading & correlation matrix generation
├── 02_index_weight_processing.ipynb     # Index weight formatting & AMPL output processing
├── 03_portfolio_analysis.ipynb          # Interactive end-to-end analysis notebook
│
├── data/                                # All input data
│   ├── weekly_returns.xlsx              # Weekly returns, 267 weeks, 97 tickers
│   ├── index_composition.xlsx           # NASDAQ-100 ticker weights (ordered DataFrame)
│   ├── sector_industry_matrix.xlsx      # 97×23 binary sector/industry membership
│   ├── correlation_matrices/            # 97×97 Pearson correlation matrices
│   │   ├── correlation_matrix_1.xlsx    # Window 1: weeks 0–185
│   │   ├── correlation_matrix_2.xlsx    # Window 2: weeks 20–205
│   │   ├── correlation_matrix_3.xlsx    # Window 3: weeks 40–225
│   │   └── correlation_matrix_4.xlsx    # Window 4: weeks 60–245
│   └── ampl_results/
│       └── out_of_sample_validations.xlsx  # Original AMPL solver output
│
├── ampl/                                # Original AMPL model files
│   ├── max_similarity.mod               # MIP model definition
│   ├── instance_1.dat                   # Data file for instance 1
│   ├── instance_2.dat                   # Data file for instance 2
│   ├── instance_3.dat                   # Data file for instance 3
│   └── instance_4.dat                   # Data file for instance 4
│
└── results/                             # Generated outputs (auto-created)
    ├── final_results.xlsx               # Metrics + weights + return series
    ├── optimization_results.xlsx        # Assignment matrices + fund weights
    ├── mip_results.pkl                  # Cached MIP solutions
    └── charts/
        ├── cumulative_returns.png
        ├── tracking_error.png
        ├── return_scatter.png
        ├── performance_dashboard.png
        ├── asset_stability.png
        ├── correlation_heatmaps.png
        └── selected_asset_weights.png
```

---

## Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl pulp
```

### Option 1: Run full pipeline (recommended)

```bash
cd "Passive Portfolio Management"
python run_analysis.py
```

This will:
1. Load return data and correlation matrices
2. Solve MIP for all 4 instances (or load from cache)
3. Compute performance metrics
4. Generate 7 charts → `results/*.png`
5. Export all results → `results/final_results.xlsx`
6. Print summary table

### Option 2: Solve MIP only

```bash
python solve_optimization.py
```

Outputs: `results/optimization_results.xlsx` (assignment matrices + fund weights)

### Option 3: Interactive notebook

Open `portfolio_analysis.ipynb` in Jupyter and run cells sequentially.

---

## Results

| Instance | Obj Value | TE (ann.) | Correlation | R² | OOS TE |
|----------|-----------|-----------|-------------|----|--------|
| 1 | 71.3428 | ~8.0% | ~0.942 | ~0.887 | ~9.1% |
| 2 | 70.5710 | ~8.5% | ~0.939 | ~0.882 | ~9.3% |
| 3 | 69.6260 | ~8.7% | ~0.937 | ~0.878 | ~9.5% |
| 4 | 69.8037 | ~8.5% | ~0.940 | ~0.883 | ~9.2% |
| **Avg** | | **~8.4%** | **~0.939** | **~0.883** | **~9.3%** |

### Core Assets (selected in all 4 instances)
`AMAT`, `BIIB`, `CEG`, `CTAS`, `GEHC`, `GOOG`, `MCHP`, `PDD`, `ROP`, `VRTX`, `WBD`

These 11 assets are robust representatives across all market regimes tested.

---

## Methodology Notes

- **Universe:** 97 of the original 101 NASDAQ-100 tickers (removed ARM, DASH, LIN, PLTR due to data limitations)
- **Returns:** Weekly close-to-close, Apr 2020 – May 2025
- **Correlation computation:** Rolling 186-week window, Pearson correlation of weekly returns
- **Solver:** PuLP with CBC (open-source) — all 4 instances solve to near-optimality in ~1.4s each
- **Weight redistribution:** Excluded asset j's index weight is added to its representative i's weight
- **Tracking Error:** Annualised standard deviation of weekly return differences (√52 × σ)
- **Out-of-sample:** Weeks beyond the in-sample window boundary

---

## Limitations & Future Work

- The model does not account for transaction costs or liquidity constraints
- Correlation-based similarity ≠ return-based tracking error minimisation
- The equal-weight-redistribution scheme is a simplification; value-weighted methods may perform better
- Extensions: sector-constrained selection, turnover limits, L1/L2 regularisation, rolling re-optimisation

---

## References

- Beasley, J.E., Meade, N., Chang, T.J. (2003). *An evolutionary heuristic for the index tracking problem.* European Journal of Operational Research.
- Gaivoronski, A.A., et al. (2005). *Index tracking with controlled risk.* OR Letters.
- AMPL model: `AMPL files/max_sim.mod` (original formulation)
