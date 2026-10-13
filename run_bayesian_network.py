import sys
import os
import pandas as pd
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger("bayesian_analysis_output.txt")

def main():
    print("="*80, flush=True)
    print("SUSTAINABLE FASHION SURVEY - BAYESIAN NETWORK (PGM) DESIGN & INFERENCE", flush=True)
    print("="*80, flush=True)

    # 1. Load CSV
    csv_file = "PGM GOOGLE FORM RESPONSES.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!", flush=True)
        return

    df_raw = pd.read_csv(csv_file)
    print(f"Loaded {len(df_raw)} survey responses.", flush=True)

    # Clean and rename columns
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
    df_raw.columns = col_names

    # Clean whitespace in string columns
    for col in ['Purchase_Frequency', 'Monthly_Spending', 'Purchase_Channel', 'Behaviour']:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].astype(str).str.strip()

    # Convert numeric Likert items (1-5)
    likert_cols = ['Awareness', 'Attitude', 'Social_Influence', 'PBC', 
                   'Price_Sensitivity', 'Trust', 'Intention', 
                   'Sustainable_Preference', 'Willingness_To_Pay']
    for col in likert_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # -------------------------------------------------------------
    # EDA & VARIABLE ANALYSIS
    # -------------------------------------------------------------
    print("\n--- 1. EXPLORATORY DATA ANALYSIS (EDA) ---", flush=True)
    corr_ps_wtp = df_raw['Price_Sensitivity'].corr(df_raw['Willingness_To_Pay'])
    print(f"Correlation between Price Sensitivity and Willingness to Pay: {corr_ps_wtp:.4f}", flush=True)

    corr_pref_beh = df_raw[['Sustainable_Preference', 'Intention', 'Attitude']].corr()
    print("\nCorrelation Matrix (Selected Items):\n", corr_pref_beh, flush=True)

    # Merge Price Sensitivity and Willingness to Pay into a unified Economic Barrier construct:
    # Price Sensitivity: 5 means high barrier (price discourages)
    # WTP: 5 means high willingness to pay (low barrier) -> Reverse-coded WTP = 6 - WTP
    # Merged Economic Barrier = (Price_Sensitivity + (6 - Willingness_To_Pay)) / 2
    # Alternatively: Merged Price_Sensitivity_Composite: High means Price is a major constraint.
    df_raw['WTP_Reversed'] = 6 - df_raw['Willingness_To_Pay']
    df_raw['Economic_Barrier'] = (df_raw['Price_Sensitivity'] + df_raw['WTP_Reversed']) / 2.0

    print("\nPrice Sensitivity (Original Mean):", df_raw['Price_Sensitivity'].mean())
    print("Willingness To Pay (Original Mean):", df_raw['Willingness_To_Pay'].mean())
    print("Merged Economic Barrier Mean:", df_raw['Economic_Barrier'].mean())

    # Discretization:
    # Small sample (n=22): 2 states per node ('Low', 'High') is mathematically optimal to avoid empty CPT cells
    # Likert (1-5): <= 3 -> 'Low', >= 4 -> 'High'
    # Behaviour: 'None' -> 'Low', '1-2 items' / '3-5 items' / '6+ items' -> 'High' (Past active sustainable buyer)
    df_bn = pd.DataFrame()
    df_bn['Awareness'] = np.where(df_raw['Awareness'] >= 4, 'High', 'Low')
    df_bn['Attitude'] = np.where(df_raw['Attitude'] >= 4, 'High', 'Low')
    df_bn['Social_Influence'] = np.where(df_raw['Social_Influence'] >= 4, 'High', 'Low')
    df_bn['PBC'] = np.where(df_raw['PBC'] >= 4, 'High', 'Low')
    
    # For Economic Barrier: >= 3.5 is 'High' barrier (price sensitive), < 3.5 is 'Low'
    df_bn['Price_Sensitivity'] = np.where(df_raw['Economic_Barrier'] >= 3.5, 'High', 'Low')
    df_bn['Trust'] = np.where(df_raw['Trust'] >= 4, 'High', 'Low')
    df_bn['Intention'] = np.where(df_raw['Intention'] >= 4, 'High', 'Low')

    # Behaviour discretization:
    # 'None' -> 'Low'
    # '1-2 items', '3-5 items', '6+ items' -> 'High'
    df_bn['Behaviour'] = np.where(df_raw['Behaviour'].str.lower().str.contains('none'), 'Low', 'High')

    print("\n--- Discretized Dataset Summary (n=22) ---", flush=True)
    for col in df_bn.columns:
        counts = df_bn[col].value_counts().to_dict()
        print(f"Node {col:18s}: {counts}")

    # -------------------------------------------------------------
    # BAYESIAN NETWORK STRUCTURE DEFINITION (DAG)
    # -------------------------------------------------------------
    # 8 Nodes:
    # Awareness -> Attitude -> Intention -> Behaviour
    # Social_Influence -> Intention
    # PBC -> Intention
    # Price_Sensitivity -> Intention
    # Trust -> Intention

    nodes = ['Awareness', 'Attitude', 'Social_Influence', 'PBC', 'Price_Sensitivity', 'Trust', 'Intention', 'Behaviour']
    states = {'Low': 0, 'High': 1}
    state_names = ['Low', 'High']

    # Parents definition
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

    # -------------------------------------------------------------
    # PARAMETER ESTIMATION (CPTs with Laplace Smoothing Dirichlet Prior alpha=1)
    # -------------------------------------------------------------
    cpts = {}
    alpha = 1.0  # Laplace smoothing pseudo-count

    for node in nodes:
        par_list = parents[node]
        if not par_list:
            # Prior node: P(Node)
            counts = {s: (df_bn[node] == s).sum() for s in state_names}
            total = len(df_bn) + 2 * alpha
            probs = {s: (counts[s] + alpha) / total for s in state_names}
            cpts[node] = probs
            print(f"\nCPT for Prior Node [{node}]: P(High)={probs['High']:.3f}, P(Low)={probs['Low']:.3f}")
        else:
            # Conditional Node: P(Node | Parents)
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
            print(f"\nCPT for Conditional Node [{node}] (Parents: {par_list}): {len(cpt_table)} parent configurations estimated.")

    # -------------------------------------------------------------
    # EXACT INFERENCE ENGINE (Joint Probability Enumeration / Variable Elimination)
    # -------------------------------------------------------------
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
        """
        Calculates P(query_vars | evidence)
        """
        hidden_vars = [v for v in nodes if v not in query_vars and v not in evidence]
        
        # Enumerate hidden variables
        q_results = {}
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

        for q_cfg, p_val in unnorm_probs.items():
            norm_val = p_val / total_p_evidence if total_p_evidence > 0 else 0.0
            q_str = ", ".join(f"{k}={v}" for k, v in zip(query_vars, q_cfg))
            q_results[q_str] = norm_val

        return q_results

    # -------------------------------------------------------------
    # RUN INFERENCE QUERIES
    # -------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("EXACT PROBABILISTIC INFERENCE QUERIES", flush=True)
    print("="*80, flush=True)

    # Base Marginals
    print("\n[PRIOR MARGINALS]:", flush=True)
    for node in nodes:
        res = query([node], {})
        print(f"P({node}): {res}")

    # Query 1:
    # P(Behaviour = High | Awareness = High, Attitude = High, Intention = High)
    q1 = query(['Behaviour'], {'Awareness': 'High', 'Attitude': 'High', 'Intention': 'High'})
    print("\n[QUERY 1] P(Behaviour | Awareness=High, Attitude=High, Intention=High):")
    for k, v in q1.items():
        print(f"  {k} = {v:.4f} ({v*100:.2f}%)")

    # Comparison for Query 1: Low Intention
    q1_low = query(['Behaviour'], {'Awareness': 'High', 'Attitude': 'High', 'Intention': 'Low'})
    print("  [Contrast with Intention=Low]:", q1_low)

    # Query 2:
    # P(Intention = High | Trust = High, PBC = High, Social Influence = High)
    q2 = query(['Intention'], {'Trust': 'High', 'PBC': 'High', 'Social_Influence': 'High'})
    print("\n[QUERY 2] P(Intention | Trust=High, PBC=High, Social_Influence=High):")
    for k, v in q2.items():
        print(f"  {k} = {v:.4f} ({v*100:.2f}%)")

    # Comparison for Query 2: All Low
    q2_low = query(['Intention'], {'Trust': 'Low', 'PBC': 'Low', 'Social_Influence': 'Low'})
    print("  [Contrast with Trust=Low, PBC=Low, Social_Influence=Low]:", q2_low)

    # Additional Query 3: Diagnostic / Abductive Inference
    # What was the likely Awareness and Attitude given that someone exhibited High Behaviour?
    q3_att = query(['Attitude'], {'Behaviour': 'High'})
    q3_awa = query(['Awareness'], {'Behaviour': 'High'})
    print("\n[QUERY 3 - Diagnostic Reasoning]: Given Behaviour=High:")
    print(f"  P(Attitude=High | Behaviour=High) = {q3_att.get('Attitude=High', 0):.4f}")
    print(f"  P(Awareness=High | Behaviour=High) = {q3_awa.get('Awareness=High', 0):.4f}")

    # Additional Query 4: Impact of Price Barrier
    q4_price_high = query(['Intention'], {'Price_Sensitivity': 'High', 'Attitude': 'High', 'Trust': 'High'})
    q4_price_low = query(['Intention'], {'Price_Sensitivity': 'Low', 'Attitude': 'High', 'Trust': 'High'})
    print("\n[QUERY 4 - Impact of Price Sensitivity on Intention]:")
    print(f"  P(Intention=High | Price_Barrier=High, Attitude=High, Trust=High) = {q4_price_high.get('Intention=High', 0):.4f}")
    print(f"  P(Intention=High | Price_Barrier=Low, Attitude=High, Trust=High)  = {q4_price_low.get('Intention=High', 0):.4f}")

    # -------------------------------------------------------------
    # GENERATE VISUALIZATIONS
    # -------------------------------------------------------------
    print("\nGenerating visual diagrams and plots...", flush=True)

    # 1. Structure Graph & Marginals
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#ffffff')

    # Subplot 1: Bar chart of Queries
    queries_labels = [
        'P(Beh=High | High Aware, Att, Int)',
        'P(Beh=High | High Aware, Att, Low Int)',
        'P(Int=High | High Trust, PBC, Social)',
        'P(Int=High | Low Trust, PBC, Social)',
        'P(Int=High | Low Price Barrier, Att, Trust)',
        'P(Int=High | High Price Barrier, Att, Trust)'
    ]
    queries_vals = [
        q1['Behaviour=High'],
        q1_low['Behaviour=High'],
        q2['Intention=High'],
        q2_low['Intention=High'],
        q4_price_low['Intention=High'],
        q4_price_high['Intention=High']
    ]
    colors = ['#2e7d32', '#c62828', '#1565c0', '#e65100', '#00838f', '#ad1457']

    y_pos = np.arange(len(queries_labels))
    axes[0].barh(y_pos, queries_vals, color=colors, height=0.55, edgecolor='black', alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(queries_labels, fontsize=10, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel('Posterior Probability', fontsize=12, fontweight='bold')
    axes[0].set_title('Bayesian Inference Query Results', fontsize=13, fontweight='bold', pad=12)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    for i, v in enumerate(queries_vals):
        axes[0].text(v + 0.02, i, f"{v:.1%}", va='center', fontweight='bold', fontsize=10)

    # Subplot 2: Prior Marginal Probabilities of High State across Nodes
    node_priors = [query([n], {})[f"{n}=High"] for n in nodes]
    node_names_display = ['Awareness', 'Attitude', 'Social Influ.', 'PBC', 'Price Barrier', 'Trust', 'Intention', 'Behaviour']
    
    bars = axes[1].bar(node_names_display, node_priors, color='#3949ab', width=0.5, edgecolor='black', alpha=0.85)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('Prior Probability P(Node = High)', fontsize=12, fontweight='bold')
    axes[1].set_title('Prior Node Marginals P(State = High)', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xticklabels(node_names_display, rotation=35, ha='right', fontsize=10, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.1%}", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('bayesian_inference_results.png', dpi=300)
    plt.close()
    print("Saved inference chart to 'bayesian_inference_results.png'.", flush=True)

    # -------------------------------------------------------------
    # EXPORT SUMMARY JSON / CSV
    # -------------------------------------------------------------
    summary_df = pd.DataFrame([
        {'Query': 'P(Behaviour=High | Awareness=High, Attitude=High, Intention=High)', 'Probability': q1['Behaviour=High']},
        {'Query': 'P(Behaviour=High | Awareness=High, Attitude=High, Intention=Low)', 'Probability': q1_low['Behaviour=High']},
        {'Query': 'P(Intention=High | Trust=High, PBC=High, Social_Influence=High)', 'Probability': q2['Intention=High']},
        {'Query': 'P(Intention=High | Trust=Low, PBC=Low, Social_Influence=Low)', 'Probability': q2_low['Intention=High']},
        {'Query': 'P(Intention=High | Price_Barrier=Low, Attitude=High, Trust=High)', 'Probability': q4_price_low['Intention=High']},
        {'Query': 'P(Intention=High | Price_Barrier=High, Attitude=High, Trust=High)', 'Probability': q4_price_high['Intention=High']},
        {'Query': 'P(Attitude=High | Behaviour=High)', 'Probability': q3_att['Attitude=High']},
        {'Query': 'P(Awareness=High | Behaviour=High)', 'Probability': q3_awa['Awareness=High']}
    ])
    summary_df.to_csv('inference_results_summary.csv', index=False)
    print("Exported results summary to 'inference_results_summary.csv'.", flush=True)

if __name__ == '__main__':
    main()
