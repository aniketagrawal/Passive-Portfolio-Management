"""
Passive Portfolio Management - Index Replication Optimizer
==========================================================
Solves a Mixed Integer Program (MIP) to select q=25 assets from the NASDAQ-100
universe that best replicate the full 97-asset index using correlation-based similarity.

Model (max_sim.mod):
  maximize  sum_{i,j} rho[i,j] * X[i,j]
  s.t.      sum_i Y[i] <= q
            X[i,j] <= Y[i]   for all i, j
            sum_i X[i,j] = 1  for all j
            X[i,j], Y[i] in {0,1}

X[i,j] = 1  =>  selected asset i represents original asset j
Y[i]   = 1  =>  asset i is included in the simplified fund

Usage:
  python solve_optimization.py
Outputs:
  results/optimization_results.xlsx  (assignment matrices + selected assets)
"""

import pandas as pd
import numpy as np
import pulp
import os
import time

# ── Configuration ─────────────────────────────────────────────────────────────
Q = 25          # max assets in simplified fund
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORR_DIR = os.path.join(BASE_DIR, "data", "correlation_matrices")
OUTPUT_FILE = os.path.join(BASE_DIR, "results", "optimization_results.xlsx")

INSTANCES = {
    1: os.path.join(CORR_DIR, "correlation_matrix_1.xlsx"),
    2: os.path.join(CORR_DIR, "correlation_matrix_2.xlsx"),
    3: os.path.join(CORR_DIR, "correlation_matrix_3.xlsx"),
    4: os.path.join(CORR_DIR, "correlation_matrix_4.xlsx"),
}

# Ordered ticker list (matches correlation matrices)
TICKERS = [
    "ADBE", "AMD", "ABNB", "AAPL", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMGN",
    "AMZN", "ANSS", "APP", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST",
    "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC", "INTU",
    "ISRG", "KDP", "KHC", "KLAC", "LRCX", "LULU", "MAR", "MCHP", "MDB", "MDLZ",
    "MELI", "META", "MNST", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA", "NXPI",
    "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PYPL", "QCOM",
    "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO",
    "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
]

# Original NASDAQ-100 index weights
INDEX_WEIGHTS = {
    "ADBE": 0.01304679819, "AMD": 0.00744483689,  "ABNB": 0.00400517720,
    "GOOG": 0.08536828073, "GOOGL": 0.08536828073, "AMZN": 0.08596606561,
    "AEP":  0.00225224054, "AMGN": 0.00826154825,  "ADI":  0.00431575390,
    "ANSS": 0.00163849524, "AAPL": 0.13868991740,  "AMAT": 0.00567491924,
    "APP":  0.00061404515, "ASML": 0.01417822079,  "AZN":  0.00909939482,
    "TEAM": 0.00340925304, "ADSK": 0.00314899054,  "ADP":  0.00515582713,
    "AXON": 0.00058945984, "BKR":  0.00152764135,  "BIIB": 0.00251865237,
    "BKNG": 0.00522546037, "AVGO": 0.01254849386,  "CDNS": 0.00253245984,
    "CDW":  0.00136681363, "CHTR": 0.00656201727,  "CTAS": 0.00239142423,
    "CSCO": 0.01243783895, "CCEP": 0.00082383762,  "CTSH": 0.00224141364,
    "CMCSA":0.01231033781, "CEG":  0.00070201548,  "CPRT": 0.00179231594,
    "CSGP": 0.00188189538, "COST": 0.01190570651,  "CRWD": 0.00228544159,
    "CSX":  0.00408221271, "DDOG": 0.00187921529,  "DXCM": 0.00248593290,
    "FANG": 0.00104258277, "EA":   0.00226233801,  "EXC":  0.00259299966,
    "FAST": 0.00177576759, "FTNT": 0.00233050075,  "GEHC": 0.00037727298,
    "GILD": 0.00546211627, "GFS":  0.00098517121,  "HON":  0.00525541555,
    "IDXX": 0.00237994668, "INTC": 0.01164038601,  "INTU": 0.00701368376,
    "ISRG": 0.00561425515, "KDP":  0.00246384909,  "KLAC": 0.00287960530,
    "KHC":  0.00274630688, "LRCX": 0.00417080725,  "LULU": 0.00265433299,
    "MAR":  0.00278483153, "MRVL": 0.00237531222,  "MELI": 0.00362340655,
    "META": 0.04090321438, "MCHP": 0.00226453548,  "MU":   0.00444197151,
    "MSFT": 0.11778347580, "MSTR": 0.00022568975,  "MDLZ": 0.00528565877,
    "MDB":  0.00116589120, "MNST": 0.00292442746,  "NFLX": 0.01142807504,
    "NVDA": 0.02835605467, "NXPI": 0.00276598739,  "ORLY": 0.00256482544,
    "ODFL": 0.00183152924, "ON":   0.00131222458,  "PCAR": 0.00195357477,
    "PANW": 0.00169402980, "PAYX": 0.00231073242,  "PYPL": 0.01106431370,
    "PDD":  0.00660689107, "PEP":  0.01347919551,  "QCOM": 0.00882966453,
    "REGN": 0.00419077002, "ROP":  0.00011811165,  "ROST": 0.00223772066,
    "SBUX": 0.00683515402, "SNPS": 0.00271136567,  "TMUS": 0.00975435576,
    "TTWO": 0.00119022390, "TSLA": 0.03817994959,  "TXN":  0.00930831127,
    "TTD":  0.00179475950, "VRSK": 0.00183785212,  "VRTX": 0.00418002831,
    "WBD":  0.00151564487, "WDAY": 0.00313075934,  "XEL":  0.00224257674,
    "ZS":   0.00142093396,
}


def load_correlation_matrix(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)
    df.columns = [str(c).split(" UW Equity")[0].split(" UQ Equity")[0].strip() for c in df.columns]
    df.index   = [str(i).split(" UW Equity")[0].split(" UQ Equity")[0].strip() for i in df.index]
    tickers_in = [t for t in TICKERS if t in df.index and t in df.columns]
    return df.loc[tickers_in, tickers_in]


def solve_instance(rho: pd.DataFrame, q: int = Q, instance_id: int = 1):
    assets = rho.index.tolist()
    n = len(assets)
    rho_np = rho.values

    print(f"\n{'='*60}")
    print(f"  Solving Instance {instance_id}  |  {n} assets  |  q={q}")
    print(f"{'='*60}")

    prob = pulp.LpProblem(f"IndexReplication_Instance{instance_id}", pulp.LpMaximize)

    Y = [pulp.LpVariable(f"Y_{i}", cat="Binary") for i in range(n)]
    X = [[pulp.LpVariable(f"X_{i}_{j}", cat="Binary") for j in range(n)] for i in range(n)]

    prob += pulp.lpSum(rho_np[i, j] * X[i][j] for i in range(n) for j in range(n))
    prob += pulp.lpSum(Y[i] for i in range(n)) <= q
    for i in range(n):
        for j in range(n):
            prob += X[i][j] <= Y[i]
    for j in range(n):
        prob += pulp.lpSum(X[i][j] for i in range(n)) == 1

    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=300, gapRel=0.005))
    elapsed = time.time() - t0

    print(f"\nStatus : {pulp.LpStatus[prob.status]}")
    print(f"Obj Val: {pulp.value(prob.objective):.6f}")
    print(f"Time   : {elapsed:.1f}s")

    Y_sol = [int(round(pulp.value(Y[i]))) for i in range(n)]
    X_sol = [[int(round(pulp.value(X[i][j]))) for j in range(n)] for i in range(n)]

    selected = [assets[i] for i in range(n) if Y_sol[i] == 1]
    print(f"Selected ({len(selected)}): {selected}")

    X_df = pd.DataFrame(X_sol, index=assets, columns=assets)
    return selected, X_df, pulp.value(prob.objective)


def compute_fund_weights(selected: list, X_df: pd.DataFrame) -> dict:
    fund_weights = {ticker: 0.0 for ticker in selected}
    for j_ticker in X_df.columns:
        col = X_df[j_ticker]
        rep = col[col > 0].index
        if len(rep) > 0:
            rep_ticker = rep[0]
            if rep_ticker in fund_weights:
                fund_weights[rep_ticker] += INDEX_WEIGHTS.get(j_ticker, 0.0)
    return fund_weights


def main():
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    all_results = {}

    for inst_id, corr_path in INSTANCES.items():
        print(f"\nLoading correlation matrix {inst_id} from:\n  {corr_path}")
        rho = load_correlation_matrix(corr_path)
        print(f"Matrix shape: {rho.shape}")

        selected, X_df, obj = solve_instance(rho, q=Q, instance_id=inst_id)
        fund_weights = compute_fund_weights(selected, X_df)

        all_results[inst_id] = {
            "selected": selected,
            "X_df": X_df,
            "obj_val": obj,
            "fund_weights": fund_weights,
        }

    print(f"\nSaving results to {OUTPUT_FILE}")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for inst_id, res in all_results.items():
            res["X_df"].to_excel(writer, sheet_name=f"Assignment_Instance{inst_id}")
            weights_df = pd.DataFrame({
                "Ticker": list(res["fund_weights"].keys()),
                "Original_Index_Weight": [INDEX_WEIGHTS.get(t, 0) for t in res["fund_weights"].keys()],
                "Fund_Weight": list(res["fund_weights"].values()),
            }).sort_values("Fund_Weight", ascending=False).reset_index(drop=True)
            weights_df.to_excel(writer, sheet_name=f"Weights_Instance{inst_id}", index=False)

            print(f"\n--- Instance {inst_id} ---")
            print(f"  Objective value : {res['obj_val']:.4f}")
            print(f"  Selected assets : {res['selected']}")
            print(f"  Fund weights sum: {sum(res['fund_weights'].values()):.4f}")

    print("\nDone! Results saved.")
    return all_results


if __name__ == "__main__":
    main()
