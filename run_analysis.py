"""
Run complete portfolio analysis and generate all outputs.
Run this script after solve_optimization.py (or let it auto-solve).
Outputs are saved to the results/ folder.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle, os, warnings
from collections import Counter

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 120, 'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11})

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = [
    'ADBE','AMD','ABNB','AAPL','ADI','ADP','ADSK','AEP','AMAT','AMGN',
    'AMZN','ANSS','APP','ASML','AVGO','AXON','AZN','BIIB','BKNG','BKR',
    'CCEP','CDNS','CDW','CEG','CHTR','CMCSA','COST','CPRT','CRWD','CSCO',
    'CSGP','CSX','CTAS','CTSH','DDOG','DXCM','EA','EXC','FANG','FAST',
    'FTNT','GEHC','GFS','GILD','GOOG','GOOGL','HON','IDXX','INTC','INTU',
    'ISRG','KDP','KHC','KLAC','LRCX','LULU','MAR','MCHP','MDB','MDLZ',
    'MELI','META','MNST','MRVL','MSFT','MSTR','MU','NFLX','NVDA','NXPI',
    'ODFL','ON','ORLY','PANW','PAYX','PCAR','PDD','PEP','PYPL','QCOM',
    'REGN','ROP','ROST','SBUX','SNPS','TEAM','TMUS','TSLA','TTD','TTWO',
    'TXN','VRSK','VRTX','WBD','WDAY','XEL','ZS',
]

INDEX_WEIGHTS = {
    'ADBE':0.01304679819,'AMD':0.00744483689,'ABNB':0.00400517720,
    'GOOG':0.08536828073,'GOOGL':0.08536828073,'AMZN':0.08596606561,
    'AEP':0.00225224054,'AMGN':0.00826154825,'ADI':0.00431575390,
    'ANSS':0.00163849524,'AAPL':0.13868991740,'AMAT':0.00567491924,
    'APP':0.00061404515,'ASML':0.01417822079,'AZN':0.00909939482,
    'TEAM':0.00340925304,'ADSK':0.00314899054,'ADP':0.00515582713,
    'AXON':0.00058945984,'BKR':0.00152764135,'BIIB':0.00251865237,
    'BKNG':0.00522546037,'AVGO':0.01254849386,'CDNS':0.00253245984,
    'CDW':0.00136681363,'CHTR':0.00656201727,'CTAS':0.00239142423,
    'CSCO':0.01243783895,'CCEP':0.00082383762,'CTSH':0.00224141364,
    'CMCSA':0.01231033781,'CEG':0.00070201548,'CPRT':0.00179231594,
    'CSGP':0.00188189538,'COST':0.01190570651,'CRWD':0.00228544159,
    'CSX':0.00408221271,'DDOG':0.00187921529,'DXCM':0.00248593290,
    'FANG':0.00104258277,'EA':0.00226233801,'EXC':0.00259299966,
    'FAST':0.00177576759,'FTNT':0.00233050075,'GEHC':0.00037727298,
    'GILD':0.00546211627,'GFS':0.00098517121,'HON':0.00525541555,
    'IDXX':0.00237994668,'INTC':0.01164038601,'INTU':0.00701368376,
    'ISRG':0.00561425515,'KDP':0.00246384909,'KLAC':0.00287960530,
    'KHC':0.00274630688,'LRCX':0.00417080725,'LULU':0.00265433299,
    'MAR':0.00278483153,'MRVL':0.00237531222,'MELI':0.00362340655,
    'META':0.04090321438,'MCHP':0.00226453548,'MU':0.00444197151,
    'MSFT':0.11778347580,'MSTR':0.00022568975,'MDLZ':0.00528565877,
    'MDB':0.00116589120,'MNST':0.00292442746,'NFLX':0.01142807504,
    'NVDA':0.02835605467,'NXPI':0.00276598739,'ORLY':0.00256482544,
    'ODFL':0.00183152924,'ON':0.00131222458,'PCAR':0.00195357477,
    'PANW':0.00169402980,'PAYX':0.00231073242,'PYPL':0.01106431370,
    'PDD':0.00660689107,'PEP':0.01347919551,'QCOM':0.00882966453,
    'REGN':0.00419077002,'ROP':0.00011811165,'ROST':0.00223772066,
    'SBUX':0.00683515402,'SNPS':0.00271136567,'TMUS':0.00975435576,
    'TTWO':0.00119022390,'TSLA':0.03817994959,'TXN':0.00930831127,
    'TTD':0.00179475950,'VRSK':0.00183785212,'VRTX':0.00418002831,
    'WBD':0.00151564487,'WDAY':0.00313075934,'XEL':0.00224257674,
    'ZS':0.00142093396,
}

WINDOWS = {1: (0, 186), 2: (20, 206), 3: (40, 226), 4: (60, 246)}
os.makedirs('results', exist_ok=True)
os.makedirs('results/charts', exist_ok=True)


def load_data():
    returns_df = pd.read_excel('data/weekly_returns.xlsx', index_col=0)
    date_mask = pd.to_datetime(returns_df.index, errors='coerce').notna()
    returns_df = returns_df[date_mask].copy()
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_clean = returns_df.fillna(0)

    corr_matrices = {}
    for inst in range(1, 5):
        df = pd.read_excel(
            f'data/correlation_matrices/correlation_matrix_{inst}.xlsx', index_col=0)
        df.columns = [str(c).split(' UW Equity')[0].split(' UQ Equity')[0].strip() for c in df.columns]
        df.index = [str(i).split(' UW Equity')[0].split(' UQ Equity')[0].strip() for i in df.index]
        t_in = [t for t in TICKERS if t in df.index]
        corr_matrices[inst] = df.loc[t_in, t_in]

    return returns_clean, corr_matrices


def get_mip_results(corr_matrices):
    CACHE = 'results/mip_results.pkl'
    if os.path.exists(CACHE):
        print('Loading cached MIP results...')
        with open(CACHE, 'rb') as f:
            return pickle.load(f)

    print('Running MIP optimization for all 4 instances...')
    import pulp, time

    def solve_mip(rho, q=25, inst_id=1):
        assets = rho.index.tolist()
        n = len(assets)
        rho_np = rho.values
        prob = pulp.LpProblem(f'MaxSim_{inst_id}', pulp.LpMaximize)
        Y = [pulp.LpVariable(f'Y_{i}', cat='Binary') for i in range(n)]
        X = [[pulp.LpVariable(f'X_{i}_{j}', cat='Binary') for j in range(n)] for i in range(n)]
        prob += pulp.lpSum(rho_np[i, j] * X[i][j] for i in range(n) for j in range(n))
        prob += pulp.lpSum(Y) <= q
        for i in range(n):
            for j in range(n):
                prob += X[i][j] <= Y[i]
        for j in range(n):
            prob += pulp.lpSum(X[i][j] for i in range(n)) == 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300, gapRel=0.01))
        Y_sol = [int(round(pulp.value(Y[i]) or 0)) for i in range(n)]
        X_sol = [[int(round(pulp.value(X[i][j]) or 0)) for j in range(n)] for i in range(n)]
        selected = [assets[i] for i in range(n) if Y_sol[i] == 1]
        X_df = pd.DataFrame(X_sol, index=assets, columns=assets)
        print(f'  Inst {inst_id}: obj={pulp.value(prob.objective):.4f}, selected={len(selected)}')
        return selected, X_df, pulp.value(prob.objective)

    def fund_weights(selected, X_df):
        fw = {t: 0.0 for t in selected}
        for j in X_df.columns:
            col = X_df[j]
            reps = col[col > 0].index
            if len(reps) > 0 and reps[0] in fw:
                fw[reps[0]] += INDEX_WEIGHTS.get(j, 0.0)
        return fw

    mip_results = {}
    for inst, rho in corr_matrices.items():
        sel, X_df, obj = solve_mip(rho, q=25, inst_id=inst)
        mip_results[inst] = {
            'selected': sel, 'X_df': X_df, 'obj_val': obj,
            'fund_weights': fund_weights(sel, X_df),
        }
    with open(CACHE, 'wb') as f:
        pickle.dump(mip_results, f)
    print('Optimization done, results cached.')
    return mip_results


def compute_returns(returns_clean, mip_results):
    w_series = pd.Series(INDEX_WEIGHTS)
    common = [c for c in returns_clean.columns if c in w_series.index]
    w_aligned = w_series[common] / w_series[common].sum()
    index_returns = (returns_clean[common] * w_aligned).sum(axis=1).iloc[1:]
    index_returns.name = 'NASDAQ-100'

    portfolio_returns = {}
    for inst, res in mip_results.items():
        fw = res['fund_weights']
        valid = [t for t in fw if t in returns_clean.columns]
        w_fund = pd.Series({t: fw[t] for t in valid})
        w_fund = w_fund / w_fund.sum()
        port_ret = (returns_clean[valid] * w_fund).sum(axis=1).iloc[1:]
        port_ret.name = f'Fund {inst}'
        portfolio_returns[inst] = port_ret

    return index_returns, portfolio_returns


def compute_metrics(index_returns, portfolio_returns):
    def mdd(r):
        w = (1 + r).cumprod()
        rm = w.cummax()
        return ((w - rm) / rm).min()

    all_m, ret_series = {}, {}
    for inst, port_ret in portfolio_returns.items():
        idx = index_returns.index.intersection(port_ret.index)
        ix, px = index_returns.loc[idx], port_ret.loc[idx]
        diff = px - ix
        te = diff.std() * np.sqrt(52)
        corr = ix.corr(px)
        n_years = len(ix) / 52
        cum_ix = (1 + ix).cumprod() - 1
        cum_px = (1 + px).cumprod() - 1
        s, e = WINDOWS[inst]
        split = min(e - 2, len(ix) - 1)
        diff_oos = diff.iloc[split:]
        all_m[inst] = {
            'TE_ann': te, 'Corr': corr, 'R2': corr ** 2,
            'AnnRet_Idx': (1 + cum_ix.iloc[-1]) ** (1 / n_years) - 1,
            'AnnRet_Fund': (1 + cum_px.iloc[-1]) ** (1 / n_years) - 1,
            'Sharpe_Idx': (ix.mean() / ix.std()) * np.sqrt(52),
            'Sharpe_Fund': (px.mean() / px.std()) * np.sqrt(52),
            'MaxDD_Idx': mdd(ix), 'MaxDD_Fund': mdd(px),
            'CumRet_Idx': cum_ix.iloc[-1], 'CumRet_Fund': cum_px.iloc[-1],
            'TE_OOS': diff_oos.std() * np.sqrt(52),
            'Corr_OOS': ix.iloc[split:].corr(px.iloc[split:]),
            'N_OOS': len(diff_oos),
        }
        ret_series[inst] = (ix, px)
    return pd.DataFrame(all_m).T, ret_series


def plot_cumulative_returns(mdf, ret_series):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for idx, (inst, (ix, px)) in enumerate(ret_series.items()):
        ax = axes[idx // 2][idx % 2]
        cum_ix = (1 + ix).cumprod()
        cum_px = (1 + px).cumprod()
        s, e = WINDOWS[inst]
        split_date = ix.index[min(e - 2, len(ix) - 1)]
        ax.plot(cum_ix.index, cum_ix.values, label='NASDAQ-100 (97 assets)', color='#2c3e50', lw=2)
        ax.plot(cum_px.index, cum_px.values, label='Simplified Fund (25 assets)',
                color='#e74c3c', lw=2, ls='--')
        ax.axvspan(ix.index[0], split_date, alpha=0.06, color='blue')
        ax.axvline(split_date, color='navy', ls=':', lw=1.5, label='In/OOS boundary')
        m = mdf.loc[inst]
        ax.set_title(f'Instance {inst}: TE={m["TE_ann"]*100:.2f}%  Corr={m["Corr"]:.4f}', fontsize=11)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Wealth ($1 invested)')
        ax.legend(fontsize=9)
    plt.suptitle('NASDAQ-100 Index vs 25-Asset Simplified Fund — All 4 Rolling Windows',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/charts/cumulative_returns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/cumulative_returns.png')


def plot_tracking_error(mdf, ret_series):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    for idx, (inst, (ix, px)) in enumerate(ret_series.items()):
        ax = axes[idx // 2][idx % 2]
        diff = px - ix
        rte = diff.rolling(26).std() * np.sqrt(52) * 100
        s, e = WINDOWS[inst]
        split_date = ix.index[min(e - 2, len(ix) - 1)]
        ax.plot(rte.index, rte.values, color='#c0392b', lw=1.8, label='Rolling 26-wk TE (ann.)')
        ax.axvline(split_date, color='navy', ls=':', lw=1.5, label='In/OOS boundary')
        overall = diff.std() * np.sqrt(52) * 100
        ax.axhline(overall, color='gray', ls='--', alpha=0.7, label=f'Overall TE={overall:.2f}%')
        ax.set_title(f'Instance {inst}: Rolling Tracking Error')
        ax.set_ylabel('Ann TE (%)')
        ax.legend(fontsize=9)
    plt.suptitle('Rolling 26-Week Tracking Error (Annualised)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/charts/tracking_error.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/tracking_error.png')


def plot_return_scatter(ret_series):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (inst, (ix, px)) in enumerate(ret_series.items()):
        ax = axes[idx // 2][idx % 2]
        ix_f, px_f = ix.astype(float), px.astype(float)
        corr = ix_f.corr(px_f)
        ax.scatter(ix_f * 100, px_f * 100, alpha=0.35, s=15, color='#3498db', edgecolors='none')
        lo = min(ix_f.min(), px_f.min()) * 100
        hi = max(ix_f.max(), px_f.max()) * 100
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect tracking')
        m_ols, b_ols = np.polyfit(ix_f * 100, px_f * 100, 1)
        xl = np.linspace(lo, hi, 100)
        ax.plot(xl, m_ols * xl + b_ols, 'g-', lw=1.5, alpha=0.8, label=f'OLS beta={m_ols:.3f}')
        ax.set_xlabel('Index Return (%)')
        ax.set_ylabel('Fund Return (%)')
        ax.set_title(f'Instance {inst}: Corr={corr:.4f}  R2={corr**2:.4f}')
        ax.legend(fontsize=9)
    plt.suptitle('Weekly Returns: Fund vs Index', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/charts/return_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/return_scatter.png')


def plot_performance_dashboard(mdf, ret_series):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    instances = list(ret_series.keys())
    colors_inst = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22']
    x = np.arange(len(instances))

    ax = axes[0, 0]
    ax.bar(x - 0.2, [mdf.loc[i, 'AnnRet_Idx'] * 100 for i in instances], 0.35,
           label='Index', color='#2c3e50', alpha=0.85)
    ax.bar(x + 0.2, [mdf.loc[i, 'AnnRet_Fund'] * 100 for i in instances], 0.35,
           label='Fund', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f'Inst {i}' for i in instances])
    ax.set_ylabel('Ann. Return (%)'); ax.set_title('Annualised Returns'); ax.legend()

    ax = axes[0, 1]
    te_vals = [mdf.loc[i, 'TE_ann'] * 100 for i in instances]
    ax.bar(x, te_vals, color=colors_inst, alpha=0.85, edgecolor='white')
    for xi, v in zip(x, te_vals):
        ax.text(xi, v + 0.05, f'{v:.2f}%', ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f'Inst {i}' for i in instances])
    ax.set_ylabel('TE (%)'); ax.set_title('Annualised Tracking Error')

    ax = axes[0, 2]
    corr_vals = [mdf.loc[i, 'Corr'] for i in instances]
    ax.bar(x, corr_vals, color=colors_inst, alpha=0.85, edgecolor='white')
    for xi, v in zip(x, corr_vals):
        ax.text(xi, v - 0.005, f'{v:.4f}', ha='center', va='top',
                fontsize=9, color='white', fontweight='bold')
    ax.set_ylim(min(corr_vals) - 0.05, 1.0)
    ax.set_xticks(x); ax.set_xticklabels([f'Inst {i}' for i in instances])
    ax.set_ylabel('Correlation'); ax.set_title('Return Correlation (Index vs Fund)')

    ax = axes[1, 0]
    ax.bar(x - 0.2, [mdf.loc[i, 'Sharpe_Idx'] for i in instances], 0.35,
           label='Index', color='#2c3e50', alpha=0.85)
    ax.bar(x + 0.2, [mdf.loc[i, 'Sharpe_Fund'] for i in instances], 0.35,
           label='Fund', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f'Inst {i}' for i in instances])
    ax.set_ylabel('Sharpe Ratio'); ax.set_title('Sharpe Ratio Comparison'); ax.legend()

    ax = axes[1, 1]
    ax.bar(x - 0.2, [mdf.loc[i, 'MaxDD_Idx'] * 100 for i in instances], 0.35,
           label='Index', color='#2c3e50', alpha=0.85)
    ax.bar(x + 0.2, [mdf.loc[i, 'MaxDD_Fund'] * 100 for i in instances], 0.35,
           label='Fund', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f'Inst {i}' for i in instances])
    ax.set_ylabel('Max DD (%)'); ax.set_title('Maximum Drawdown'); ax.legend()

    ax = axes[1, 2]
    te_is, te_oos = [], []
    for inst, (ix, px) in ret_series.items():
        s, e = WINDOWS[inst]
        split = min(e - 2, len(ix) - 1)
        te_is.append((px - ix).iloc[:split].std() * np.sqrt(52) * 100)
        te_oos.append(mdf.loc[inst, 'TE_OOS'] * 100)
    ax.bar(x - 0.2, te_is, 0.35, label='In-sample', color='#3498db', alpha=0.85)
    ax.bar(x + 0.2, te_oos, 0.35, label='Out-of-sample', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f'Inst {i}' for i in instances])
    ax.set_ylabel('Ann TE (%)'); ax.set_title('In vs Out-of-Sample TE'); ax.legend()

    plt.suptitle('Performance Dashboard: 25-Asset Fund vs NASDAQ-100',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/charts/performance_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/performance_dashboard.png')


def plot_asset_stability(mip_results):
    all_sel = [t for res in mip_results.values() for t in res['selected']]
    freq = Counter(all_sel)
    freq_df = pd.DataFrame(freq.most_common(), columns=['Ticker', 'Count'])
    color_map = {4: '#2ecc71', 3: '#3498db', 2: '#f39c12', 1: '#e74c3c'}
    bar_colors = [color_map[c] for c in freq_df['Count']]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(freq_df)), freq_df['Count'], color=bar_colors, edgecolor='white')
    ax.set_xticks(range(len(freq_df)))
    ax.set_xticklabels(freq_df['Ticker'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Times Selected (out of 4)')
    ax.set_title('Asset Stability Across Market Regimes', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 4.5)
    patches = [
        mpatches.Patch(color=color_map[k], label=v)
        for k, v in {4: 'Core (4/4)', 3: 'Stable (3/4)',
                     2: 'Moderate (2/4)', 1: 'Occasional (1/4)'}.items()
    ]
    ax.legend(handles=patches, loc='upper right')
    plt.tight_layout()
    plt.savefig('results/charts/asset_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/asset_stability.png')
    return freq


def plot_correlation_heatmaps(mip_results, corr_matrices):
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    for idx, (inst, res) in enumerate(mip_results.items()):
        ax = axes[idx // 2][idx % 2]
        sel = res['selected']
        rho = corr_matrices[inst]
        sel_in = [t for t in sel if t in rho.index]
        sub = rho.loc[sel_in, sel_in]
        mask = np.triu(np.ones_like(sub, dtype=bool))
        sns.heatmap(sub, ax=ax, mask=mask, cmap='RdYlGn', vmin=-0.2, vmax=1,
                    annot=True, fmt='.2f', annot_kws={'size': 6},
                    xticklabels=sel_in, yticklabels=sel_in, linewidths=0.3)
        ax.set_title(f'Instance {inst}: Correlation Among 25 Selected Assets', fontsize=11)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=7)
    plt.suptitle('Pairwise Correlations of Selected Assets', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/charts/correlation_heatmaps.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: results/correlation_heatmaps.png')


def plot_fund_weights(mip_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = list(plt.cm.Set2.colors) * 10
    for idx, (inst, res) in enumerate(mip_results.items()):
        ax = axes[idx // 2][idx % 2]
        fw = res['fund_weights']
        sorted_fw = sorted(fw.items(), key=lambda x: x[1], reverse=True)
        tplot = [t for t, w in sorted_fw]
        wplot = [w * 100 for t, w in sorted_fw]
        ax.barh(range(len(tplot)), wplot, color=colors[:len(tplot)])
        ax.set_yticks(range(len(tplot)))
        ax.set_yticklabels(tplot, fontsize=8)
        ax.set_xlabel('Weight (%)')
        ax.set_title(f'Instance {inst}: Fund Weights (Obj={res["obj_val"]:.3f})')
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('results/charts/selected_asset_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/selected_asset_weights.png')


def export_excel(mdf, mip_results, ret_series):
    with pd.ExcelWriter('results/final_results.xlsx', engine='openpyxl') as writer:
        mdf.to_excel(writer, sheet_name='Performance_Metrics')
        for inst, res in mip_results.items():
            fw_df = pd.DataFrame([
                {
                    'Ticker': t,
                    'Fund_Weight': w,
                    'Index_Weight': INDEX_WEIGHTS.get(t, 0),
                    'Weight_Delta': w - INDEX_WEIGHTS.get(t, 0),
                    'Assets_Represented': int((res['X_df'].loc[t] > 0).sum())
                    if t in res['X_df'].index else 0,
                }
                for t, w in sorted(res['fund_weights'].items(), key=lambda x: x[1], reverse=True)
            ])
            fw_df.to_excel(writer, sheet_name=f'Weights_Inst{inst}', index=False)
            ix, px = ret_series[inst]
            ts_df = pd.DataFrame({
                'Index_Return': ix.values,
                'Fund_Return': px.values,
                'Tracking_Diff': (px - ix).values,
            }, index=ix.index)
            ts_df['Cum_Index'] = (1 + ts_df['Index_Return']).cumprod()
            ts_df['Cum_Fund'] = (1 + ts_df['Fund_Return']).cumprod()
            ts_df.to_excel(writer, sheet_name=f'Returns_Inst{inst}')
            res['X_df'].to_excel(writer, sheet_name=f'Assignment_Inst{inst}')
    print('Saved: results/final_results.xlsx')


def print_summary(mdf, freq):
    print()
    print('=' * 65)
    print('  RESULTS SUMMARY — NASDAQ-100 INDEX REPLICATION')
    print('=' * 65)
    print(f'  Universe : 97 NASDAQ-100 equities (Apr 2020 – May 2025)')
    print(f'  Fund size: 25 assets (74% reduction)')
    print(f'  Method   : MIP — maximise correlation-weighted similarity')
    print(f'  Instances: 4 rolling 186-week windows (20-week offset)')
    print()
    for inst in sorted(mdf.index):
        m = mdf.loc[inst]
        print(f'  Instance {inst}:')
        print(f'    Tracking Error (ann.)  : {m["TE_ann"]*100:.2f}%')
        print(f'    Correlation            : {m["Corr"]:.4f}')
        print(f'    R squared              : {m["R2"]:.4f}')
        print(f'    Ann Return  (Index)    : {m["AnnRet_Idx"]*100:.2f}%')
        print(f'    Ann Return  (Fund)     : {m["AnnRet_Fund"]*100:.2f}%')
        print(f'    Sharpe (Index / Fund)  : {m["Sharpe_Idx"]:.3f} / {m["Sharpe_Fund"]:.3f}')
        print(f'    Out-of-sample TE (ann.): {m["TE_OOS"]*100:.2f}%')
        print(f'    Out-of-sample Corr     : {m["Corr_OOS"]:.4f}  ({int(m["N_OOS"])} weeks)')
        print()
    avg_te = mdf['TE_ann'].mean() * 100
    avg_cor = mdf['Corr'].mean()
    print(f'  Average TE  : {avg_te:.2f}% p.a.')
    print(f'  Average Corr: {avg_cor:.4f}')
    core = [t for t, c in freq.most_common() if c == 4]
    print(f'  Core assets (all 4 instances): {core}')
    print()
    print('  Output files in ./results/:')
    for fn in ['final_results.xlsx', 'optimization_results.xlsx', 'mip_results.pkl',
               'charts/selected_asset_weights.png', 'charts/asset_stability.png',
               'charts/cumulative_returns.png', 'charts/tracking_error.png',
               'charts/return_scatter.png', 'charts/performance_dashboard.png',
               'charts/correlation_heatmaps.png']:
        print(f'    {fn}')


if __name__ == '__main__':
    print('Loading data...')
    returns_clean, corr_matrices = load_data()

    mip_results = get_mip_results(corr_matrices)

    print('Computing returns...')
    index_returns, portfolio_returns = compute_returns(returns_clean, mip_results)

    print('Computing metrics...')
    mdf, ret_series = compute_metrics(index_returns, portfolio_returns)

    print('Generating plots...')
    plot_cumulative_returns(mdf, ret_series)
    plot_tracking_error(mdf, ret_series)
    plot_return_scatter(ret_series)
    plot_performance_dashboard(mdf, ret_series)
    freq = plot_asset_stability(mip_results)
    plot_correlation_heatmaps(mip_results, corr_matrices)
    plot_fund_weights(mip_results)
    export_excel(mdf, mip_results, ret_series)
    print_summary(mdf, freq)
