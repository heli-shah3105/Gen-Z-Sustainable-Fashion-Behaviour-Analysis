import os
import itertools
import pandas as pd
import numpy as np

# Set print options
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

csv_path = "PGM GOOGLE FORM RESPONSES.csv"
df_raw = pd.read_csv(csv_path)

print(f"Total Rows Loaded: {len(df_raw)}")
print("\nRaw Column Names in CSV:")
for i, col in enumerate(df_raw.columns):
    print(f"  Col {i:2d}: {repr(col)}")

# Clean column headers
col_names = [
    'Timestamp',
    'Purchase_Frequency',
    'Monthly_Spending',
    'Purchase_Channel',
    'Awareness',
    'Attitude',
    'Social_Influence',
    'PBC',
    'Price_Sensitivity',
    'Trust',
    'Intention',
    'Sustainable_Preference',
    'Willingness_To_Pay',
    'Behaviour'
]

df = df_raw.copy()
df.columns = col_names

# Strip strings
for col in ['Purchase_Frequency', 'Monthly_Spending', 'Purchase_Channel', 'Behaviour']:
    df[col] = df[col].astype(str).str.strip()

# Print Behaviour counts
print("\nRaw Behaviour value counts:")
print(df['Behaviour'].value_counts())

# Numeric conversions
likert_cols = ['Awareness', 'Attitude', 'Social_Influence', 'PBC', 
               'Price_Sensitivity', 'Trust', 'Intention', 
               'Sustainable_Preference', 'Willingness_To_Pay']
for col in likert_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n--- Descriptive Statistics of Likert Items ---")
print(df[likert_cols].describe().round(3).T[['count', 'mean', 'std', 'min', '50%', 'max']])

# -------------------------------------------------------------------------
# ANALYSIS OF EXTRA QUESTIONS: WTP vs Price Sensitivity & Sustainable Preference
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("ANALYSIS OF EXTRA QUESTIONS & CORRELATIONS")
print("="*80)

corr_matrix = df[likert_cols].corr()
print("\nFull Correlation Matrix:\n", corr_matrix.round(3))

r_ps_wtp = df['Price_Sensitivity'].corr(df['Willingness_To_Pay'])
print(f"\n1. Price Sensitivity vs Willingness to Pay Correlation: r = {r_ps_wtp:.3f}")
print("   - Price Sensitivity item: 'A significant price difference discourages me...' (Barrier/Negative)")
print("   - Willingness to Pay item: 'I am willing to pay slightly more...' (Enabler/Positive)")
print("   - Semantic alignment: Merged Economic Barrier Index = [Price_Sensitivity + (6 - WTP)] / 2")

df['Economic_Barrier'] = (df['Price_Sensitivity'] + (6 - df['Willingness_To_Pay'])) / 2.0
print(f"   - Merged Economic Barrier Index Mean: {df['Economic_Barrier'].mean():.2f} (SD: {df['Economic_Barrier'].std():.2f})")

# Sustainable Preference vs Past Behaviour
print(f"\n2. Sustainable Preference ('When similar options are available, I prefer...'):")
print(f"   - Correlation with Attitude: r = {df['Sustainable_Preference'].corr(df['Attitude']):.3f}")
print(f"   - Correlation with Intention: r = {df['Sustainable_Preference'].corr(df['Intention']):.3f}")
print("   - Note: Preference represents an evaluative/affective tendency, distinct from actual past purchases.")

# -------------------------------------------------------------------------
# DISCRETIZATION (for Small-Sample n=22 Robust PGM Inference)
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("DISCRETIZATION FOR BAYESIAN NETWORK (2-State Representation: Low vs High)")
print("="*80)

df_bn = pd.DataFrame()
df_bn['Awareness'] = np.where(df['Awareness'] >= 4, 'High', 'Low')
df_bn['Attitude'] = np.where(df['Attitude'] >= 4, 'High', 'Low')
df_bn['Social_Influence'] = np.where(df['Social_Influence'] >= 4, 'High', 'Low')
df_bn['PBC'] = np.where(df['PBC'] >= 4, 'High', 'Low')
df_bn['Price_Sensitivity'] = np.where(df['Economic_Barrier'] >= 3.5, 'High', 'Low')
df_bn['Trust'] = np.where(df['Trust'] >= 4, 'High', 'Low')
df_bn['Intention'] = np.where(df['Intention'] >= 4, 'High', 'Low')

# Behaviour: None -> Low; 1-2 items / 3-5 items / 6+ items -> High
df_bn['Behaviour'] = np.where(df['Behaviour'].str.lower() == 'none', 'Low', 'High')

print("\nDiscretized State Frequencies (n=22):")
for col in df_bn.columns:
    vc = df_bn[col].value_counts().to_dict()
    pct_high = (df_bn[col] == 'High').mean() * 100
    print(f"  {col:18s}: High={vc.get('High', 0):2d} ({pct_high:5.1f}%), Low={vc.get('Low', 0):2d} ({100-pct_high:5.1f}%)")

# -------------------------------------------------------------------------
# CORE BAYESIAN NETWORK DEFINITION (8 NODES)
# -------------------------------------------------------------------------
nodes = ['Awareness', 'Attitude', 'Social_Influence', 'PBC', 'Price_Sensitivity', 'Trust', 'Intention', 'Behaviour']
state_names = ['Low', 'High']

parents = {
    'Awareness': [],
    'Attitude': ['Awareness'],
    'Social_Influence': [],
    'PBC': [],
    'Price_Sensitivity': [],
    'Trust': [],
    'Intention': ['Attitude', 'Social_Influence', 'PBC', 'Price_Sensitivity', 'Trust'],
    'Behaviour': ['Intention']
}

# -------------------------------------------------------------------------
# PARAMETER ESTIMATION (CPTs with Laplace Smoothing Dirichlet Prior alpha=1)
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("CONDITIONAL PROBABILITY TABLES (CPTs) WITH LAPLACE SMOOTHING (alpha=1)")
print("="*80)

cpts = {}
alpha = 1.0

for node in nodes:
    par_list = parents[node]
    if not par_list:
        counts = {s: (df_bn[node] == s).sum() for s in state_names}
        total = len(df_bn) + 2 * alpha
        probs = {s: (counts[s] + alpha) / total for s in state_names}
        cpts[node] = probs
        print(f"\n[Prior Node] P({node}):")
        for s in state_names:
            print(f"  P({node}={s}) = {probs[s]:.4f} (Count: {counts[s]})")
    else:
        par_configs = list(itertools.product(state_names, repeat=len(par_list)))
        cpt_table = {}
        for cfg in par_configs:
            cond = pd.Series(True, index=df_bn.index)
            for p_name, p_val in zip(par_list, cfg):
                cond = cond & (df_bn[p_name] == p_val)
            n_matching = cond.sum()
            counts = {s: (cond & (df_bn[node] == s)).sum() for s in state_names}
            total = n_matching + 2 * alpha
            probs = {s: (counts[s] + alpha) / total for s in state_names}
            cpt_table[cfg] = probs
        cpts[node] = cpt_table
        print(f"\n[Conditional Node] P({node} | {', '.join(par_list)}) - ({len(cpt_table)} configurations estimated):")
        # Print sample configurations
        if len(par_list) == 1:
            for cfg, probs in cpt_table.items():
                print(f"  P({node}=High | {par_list[0]}={cfg[0]}) = {probs['High']:.4f}, P(Low) = {probs['Low']:.4f}")

# -------------------------------------------------------------------------
# EXACT INFERENCE ENGINE
# -------------------------------------------------------------------------
def get_node_prob(node, node_val, evidence_dict):
    par_list = parents[node]
    if not par_list:
        return cpts[node][node_val]
    else:
        cfg = tuple(evidence_dict[p] for p in par_list)
        return cpts[node][cfg][node_val]

def joint_probability(assignment):
    p = 1.0
    for node in nodes:
        p *= get_node_prob(node, assignment[node], assignment)
    return p

def query(query_vars, evidence={}):
    hidden_vars = [v for v in nodes if v not in query_vars and v not in evidence]
    q_configs = list(itertools.product(state_names, repeat=len(query_vars)))
    
    total_p_evidence = 0.0
    unnorm_probs = {}

    for q_cfg in q_configs:
        q_dict = dict(zip(query_vars, q_cfg))
        base_assignment = {**evidence, **q_dict}
        
        p_sum = 0.0
        h_configs = list(itertools.product(state_names, repeat=len(hidden_vars)))
        for h_cfg in h_configs:
            h_dict = dict(zip(hidden_vars, h_cfg))
            full_assign = {**base_assignment, **h_dict}
            p_sum += joint_probability(full_assign)
        
        unnorm_probs[q_cfg] = p_sum
        total_p_evidence += p_sum

    q_results = {}
    for q_cfg, p_val in unnorm_probs.items():
        norm_val = p_val / total_p_evidence if total_p_evidence > 0 else 0.0
        q_str = ", ".join(f"{k}={v}" for k, v in zip(query_vars, q_cfg))
        q_results[q_str] = norm_val

    return q_results

# -------------------------------------------------------------------------
# INFERENCE QUERIES
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("EXACT BAYESIAN NETWORK INFERENCE RESULTS")
print("="*80)

# 1. Base Prior Marginals
print("\n[1. Base Prior Marginals P(Node = High)]:")
for node in nodes:
    res = query([node], {})
    p_high = res[f"{node}=High"]
    print(f"  P({node:18s} = High) = {p_high:.4f} ({p_high*100:5.1f}%)")

# 2. Required Query 1:
# P(Behaviour = High | Awareness = High, Attitude = High, Intention = High)
q1 = query(['Behaviour'], {'Awareness': 'High', 'Attitude': 'High', 'Intention': 'High'})
q1_low_int = query(['Behaviour'], {'Awareness': 'High', 'Attitude': 'High', 'Intention': 'Low'})
print("\n[2. Required Query 1: Direct Pathway to Behaviour]")
print("  Evidence: Awareness=High, Attitude=High, Intention=High")
print(f"  --> P(Behaviour = High | Aware=H, Att=H, Int=H) = {q1['Behaviour=High']:.4f} ({q1['Behaviour=High']*100:.2f}%)")
print(f"  --> P(Behaviour = Low  | Aware=H, Att=H, Int=H) = {q1['Behaviour=Low']:.4f} ({q1['Behaviour=Low']*100:.2f}%)")
print(f"  [Contrast with Intention=Low]: P(Behaviour=High | Int=Low) = {q1_low_int['Behaviour=High']:.4f} ({q1_low_int['Behaviour=High']*100:.2f}%)")

# 3. Required Query 2:
# P(Intention = High | Trust = High, PBC = High, Social Influence = High)
q2 = query(['Intention'], {'Trust': 'High', 'PBC': 'High', 'Social_Influence': 'High'})
q2_all_low = query(['Intention'], {'Trust': 'Low', 'PBC': 'Low', 'Social_Influence': 'Low'})
print("\n[3. Required Query 2: Antecedents of Purchase Intention]")
print("  Evidence: Trust=High, PBC=High, Social_Influence=High")
print(f"  --> P(Intention = High | Trust=H, PBC=H, Soc=H) = {q2['Intention=High']:.4f} ({q2['Intention=High']*100:.2f}%)")
print(f"  --> P(Intention = Low  | Trust=H, PBC=H, Soc=H) = {q2['Intention=Low']:.4f} ({q2['Intention=Low']*100:.2f}%)")
print(f"  [Contrast with All Low Antecedents]: P(Intention=High | Trust=L, PBC=L, Soc=L) = {q2_all_low['Intention=High']:.4f} ({q2_all_low['Intention=High']*100:.2f}%)")

# 4. Additional Query: Price Barrier Impact
q_barrier_high = query(['Intention'], {'Price_Sensitivity': 'High', 'Attitude': 'High', 'Trust': 'High'})
q_barrier_low = query(['Intention'], {'Price_Sensitivity': 'Low', 'Attitude': 'High', 'Trust': 'High'})
print("\n[4. Policy / Intervention Query: Price Barrier Impact on Intention]")
print(f"  --> P(Intention=High | Price_Barrier=High, Att=High, Trust=High) = {q_barrier_high['Intention=High']:.4f}")
print(f"  --> P(Intention=High | Price_Barrier=Low,  Att=High, Trust=High) = {q_barrier_low['Intention=High']:.4f}")

# 5. Diagnostic (Abductive) Query:
# Given Behaviour=High vs Behaviour=Low, what was the underlying Awareness / Attitude?
q_diag_beh_high_att = query(['Attitude'], {'Behaviour': 'High'})
q_diag_beh_low_att  = query(['Attitude'], {'Behaviour': 'Low'})
q_diag_beh_high_awa = query(['Awareness'], {'Behaviour': 'High'})
q_diag_beh_low_awa  = query(['Awareness'], {'Behaviour': 'Low'})

print("\n[5. Diagnostic / Backward Reasoning Queries]")
print(f"  --> P(Attitude = High | Behaviour = High) = {q_diag_beh_high_att['Attitude=High']:.4f}")
print(f"  --> P(Attitude = High | Behaviour = Low)  = {q_diag_beh_low_att['Attitude=High']:.4f}")
print(f"  --> P(Awareness = High | Behaviour = High) = {q_diag_beh_high_awa['Awareness=High']:.4f}")
print(f"  --> P(Awareness = High | Behaviour = Low)  = {q_diag_beh_low_awa['Awareness=High']:.4f}")

# -------------------------------------------------------------------------
# EXPORT OUTPUT TO TEXT AND CSV
# -------------------------------------------------------------------------
output_summary = pd.DataFrame([
    {'Query_Description': 'P(Behaviour=High | Awareness=High, Attitude=High, Intention=High)', 'P_High': q1['Behaviour=High'], 'P_Low': q1['Behaviour=Low']},
    {'Query_Description': 'P(Behaviour=High | Awareness=High, Attitude=High, Intention=Low)', 'P_High': q1_low_int['Behaviour=High'], 'P_Low': q1_low_int['Behaviour=Low']},
    {'Query_Description': 'P(Intention=High | Trust=High, PBC=High, Social_Influence=High)', 'P_High': q2['Intention=High'], 'P_Low': q2['Intention=Low']},
    {'Query_Description': 'P(Intention=High | Trust=Low, PBC=Low, Social_Influence=Low)', 'P_High': q2_all_low['Intention=High'], 'P_Low': q2_all_low['Intention=Low']},
    {'Query_Description': 'P(Intention=High | Price_Barrier=High, Attitude=High, Trust=High)', 'P_High': q_barrier_high['Intention=High'], 'P_Low': q_barrier_high['Intention=Low']},
    {'Query_Description': 'P(Intention=High | Price_Barrier=Low, Attitude=High, Trust=High)', 'P_High': q_barrier_low['Intention=High'], 'P_Low': q_barrier_low['Intention=Low']},
    {'Query_Description': 'P(Attitude=High | Behaviour=High)', 'P_High': q_diag_beh_high_att['Attitude=High'], 'P_Low': q_diag_beh_high_att['Attitude=Low']},
    {'Query_Description': 'P(Attitude=High | Behaviour=Low)', 'P_High': q_diag_beh_low_att['Attitude=High'], 'P_Low': q_diag_beh_low_att['Attitude=Low']},
    {'Query_Description': 'P(Awareness=High | Behaviour=High)', 'P_High': q_diag_beh_high_awa['Awareness=High'], 'P_Low': q_diag_beh_high_awa['Awareness=Low']},
    {'Query_Description': 'P(Awareness=High | Behaviour=Low)', 'P_High': q_diag_beh_low_awa['Awareness=High'], 'P_Low': q_diag_beh_low_awa['Awareness=Low']}
])
output_summary.to_csv('final_bayesian_inference_results.csv', index=False)
print("\nSuccessfully exported final inference results to 'final_bayesian_inference_results.csv'.")
