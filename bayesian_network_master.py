import sys
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure UTF-8 output for Windows consoles and redirection
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8", errors="replace")
    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            enc = getattr(self.terminal, 'encoding', 'cp1252') or 'cp1252'
            self.terminal.write(message.encode(enc, errors='replace').decode(enc))
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    sys.stdout = DualLogger("master_output.txt")
    print("="*85)
    print(" BAYESIAN NETWORK FOR SUSTAINABLE FASHION SURVEY (PGM DESIGN & INFERENCE) ")
    print("="*85)

    # 1. Load Data
    csv_file = "PGM GOOGLE FORM RESPONSES.csv"
    df_raw = pd.read_csv(csv_file)
    print(f"Loaded {len(df_raw)} respondent records.")

    # 2. Map and Clean Column Names
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
        'Behaviour_Raw'
    ]
    df = df_raw.copy()
    df.columns = col_names

    # Clean text columns and normalize currency symbols
    for c in ['Purchase_Frequency', 'Monthly_Spending', 'Purchase_Channel', 'Behaviour_Raw']:
        df[c] = df[c].astype(str).str.strip().str.replace('₹', 'Rs. ', regex=False)

    # Convert Likert columns to numeric (1-5)
    likert_cols = ['Awareness', 'Attitude', 'Social_Influence', 'PBC', 
                   'Price_Sensitivity', 'Trust', 'Intention', 
                   'Sustainable_Preference', 'Willingness_To_Pay']
    for c in likert_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    print("\n--- Summary of Raw Behaviour Construct ---")
    print(df['Behaviour_Raw'].value_counts())

    # ---------------------------------------------------------
    # 3. EXTRA QUESTIONS ANALYSIS (WTP & Sustainable Preference)
    # ---------------------------------------------------------
    print("\n" + "="*85)
    print(" EXPLORATORY DATA ANALYSIS & EXTRA QUESTIONS ")
    print("="*85)
    
    corr_ps_wtp = df['Price_Sensitivity'].corr(df['Willingness_To_Pay'])
    print(f"\n[1] Willingness to Pay (WTP) vs Price Sensitivity:")
    print(f"    - Price Sensitivity Item: 'A significant price difference discourages me...' (Mean: {df['Price_Sensitivity'].mean():.2f})")
    print(f"    - Willingness to Pay Item: 'I am willing to pay slightly more...' (Mean: {df['Willingness_To_Pay'].mean():.2f})")
    print(f"    - Raw Pearson Correlation: r = {corr_ps_wtp:.4f}")
    print("    - Directional Reconciliation: Price Sensitivity measures an economic barrier (higher = more discouraged),")
    print("      whereas WTP measures economic tolerance (higher = more willing).")
    print("    - Reverse-coding WTP (6 - WTP) converts it into a cost barrier dimension.")
    
    df['WTP_Barrier'] = 6 - df['Willingness_To_Pay']
    df['Economic_Barrier_Index'] = (df['Price_Sensitivity'] + df['WTP_Barrier']) / 2.0
    print(f"    - Merged Economic Barrier Index Mean: {df['Economic_Barrier_Index'].mean():.2f}")

    print(f"\n[2] Sustainable Preference Item:")
    print(f"    - 'When similar options are available, I prefer the more environmentally responsible one.' (Mean: {df['Sustainable_Preference'].mean():.2f})")
    print(f"    - Correlation with Attitude: r = {df['Sustainable_Preference'].corr(df['Attitude']):.4f}")
    print(f"    - Correlation with Purchase Intention: r = {df['Sustainable_Preference'].corr(df['Intention']):.4f}")
    print("    - Theoretical Role: Measures self-reported preference (evaluative predisposition). Kept separate from actual behaviour")
    print("      to preserve the well-established 'Attitude-Intention-Behaviour Gap' in TPB literature.")

    print(f"\n[3] Excluded Variables (Reserved for EDA & Demographics):")
    print(f"    - Purchase Frequency: {df['Purchase_Frequency'].value_counts().to_dict()}")
    print(f"    - Monthly Spending: {df['Monthly_Spending'].value_counts().to_dict()}")

    # ---------------------------------------------------------
    # 4. DISCRETIZATION FOR BAYESIAN NETWORK (n=22)
    # ---------------------------------------------------------
    print("\n" + "="*85)
    print(" DISCRETIZATION (BINARY STATES: Low vs High) ")
    print("="*85)

    df_bn = pd.DataFrame()
    df_bn['Awareness'] = np.where(df['Awareness'] >= 4, 'High', 'Low')
    df_bn['Attitude'] = np.where(df['Attitude'] >= 4, 'High', 'Low')
    df_bn['Social_Influence'] = np.where(df['Social_Influence'] >= 4, 'High', 'Low')
    df_bn['PBC'] = np.where(df['PBC'] >= 4, 'High', 'Low')
    df_bn['Price_Sensitivity'] = np.where(df['Economic_Barrier_Index'] >= 3.5, 'High', 'Low')
    df_bn['Trust'] = np.where(df['Trust'] >= 4, 'High', 'Low')
    df_bn['Intention'] = np.where(df['Intention'] >= 4, 'High', 'Low')
    
    # Behaviour: None -> Low (0 purchases); 1-2, 3-5, 6+ items -> High (active sustainable buyer)
    df_bn['Behaviour'] = np.where(df['Behaviour_Raw'].str.strip() == 'None', 'Low', 'High')

    print("\nDiscretized State Counts:")
    for col in df_bn.columns:
        counts = df_bn[col].value_counts().to_dict()
        pct = (df_bn[col] == 'High').mean() * 100
        print(f"  {col:18s}: High = {counts.get('High', 0):2d} ({pct:5.1f}%), Low = {counts.get('Low', 0):2d} ({100-pct:5.1f}%)")

    # ---------------------------------------------------------
    # 5. CORE BAYESIAN NETWORK SPECIFICATION (DAG)
    # ---------------------------------------------------------
    # Nodes:
    # 1. Awareness (Root)
    # 2. Attitude (Child of Awareness)
    # 3. Social_Influence (Root)
    # 4. PBC (Root)
    # 5. Price_Sensitivity (Root - Merged Economic Barrier)
    # 6. Trust (Root)
    # 7. Intention (Child of Attitude, Social_Influence, PBC, Price_Sensitivity, Trust)
    # 8. Behaviour (Child of Intention)

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

    # ---------------------------------------------------------
    # 6. PARAMETER ESTIMATION (CPTs WITH LAPLACE SMOOTHING alpha=1)
    # ---------------------------------------------------------
    print("\n" + "="*85)
    print(" CONDITIONAL PROBABILITY TABLES (CPTs) WITH LAPLACE SMOOTHING (alpha=1) ")
    print("="*85)

    cpts = {}
    alpha = 1.0  # Laplace smoothing to handle zero-count parent combinations

    for node in nodes:
        par_list = parents[node]
        if not par_list:
            counts = {s: (df_bn[node] == s).sum() for s in state_names}
            total = len(df_bn) + 2 * alpha
            probs = {s: (counts[s] + alpha) / total for s in state_names}
            cpts[node] = probs
            print(f"\n[Prior Node] P({node}):")
            print(f"    P({node}=High) = {probs['High']:.4f}, P({node}=Low) = {probs['Low']:.4f} (Counts: High={counts['High']}, Low={counts['Low']})")
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
            print(f"\n[Conditional Node] P({node} | {', '.join(par_list)}) [{len(cpt_table)} parent states]:")
            if len(par_list) <= 2:
                for cfg, probs in cpt_table.items():
                    par_str = ", ".join(f"{p}={v}" for p, v in zip(par_list, cfg))
                    print(f"    P({node}=High | {par_str}) = {probs['High']:.4f}, P(Low) = {probs['Low']:.4f}")

    # ---------------------------------------------------------
    # 7. EXACT INFERENCE ENGINE (Variable Elimination / Exact Factor Product)
    # ---------------------------------------------------------
    def get_node_prob(node, node_val, evidence_dict):
        par_list = parents[node]
        if not par_list:
            return cpts[node][node_val]
        else:
            cfg = tuple(evidence_dict[p] for p in par_list)
            return cpts[node][cfg][node_val]

    def joint_prob(assignment):
        p = 1.0
        for node in nodes:
            p *= get_node_prob(node, assignment[node], assignment)
        return p

    def run_query(query_vars, evidence={}):
        hidden_vars = [v for v in nodes if v not in query_vars and v not in evidence]
        q_configs = list(itertools.product(state_names, repeat=len(query_vars)))
        
        unnorm = {}
        total_ev = 0.0
        
        h_configs = list(itertools.product(state_names, repeat=len(hidden_vars)))
        for q_cfg in q_configs:
            q_dict = dict(zip(query_vars, q_cfg))
            base_assign = {**evidence, **q_dict}
            
            p_sum = 0.0
            for h_cfg in h_configs:
                h_dict = dict(zip(hidden_vars, h_cfg))
                full_assign = {**base_assign, **h_dict}
                p_sum += joint_prob(full_assign)
            
            unnorm[q_cfg] = p_sum
            total_ev += p_sum

        results = {}
        for q_cfg, p_val in unnorm.items():
            norm_val = p_val / total_ev if total_ev > 0 else 0.0
            q_str = ", ".join(f"{k}={v}" for k, v in zip(query_vars, q_cfg))
            results[q_str] = norm_val
        return results

    # ---------------------------------------------------------
    # 8. EXECUTE INFERENCE QUERIES
    # ---------------------------------------------------------
    print("\n" + "="*85)
    print(" EXACT BAYESIAN NETWORK INFERENCE RESULTS ")
    print("="*85)

    # Base Marginals
    print("\n--- 1. Baseline Prior Marginals P(Node = High) ---")
    marginals = {}
    for n in nodes:
        res = run_query([n], {})
        p_h = res[f"{n}=High"]
        marginals[n] = p_h
        print(f"  P({n:18s} = High) = {p_h:.4f} ({p_h*100:5.1f}%)")

    # Required Query 1:
    # P(Behaviour = High | Awareness = High, Attitude = High, Intention = High)
    q1_high = run_query(['Behaviour'], {'Awareness': 'High', 'Attitude': 'High', 'Intention': 'High'})
    q1_low  = run_query(['Behaviour'], {'Awareness': 'High', 'Attitude': 'High', 'Intention': 'Low'})
    
    print("\n--- 2. Required Query 1: Behaviour Given Full Favourable Cognitive Chain ---")
    print("  Evidence: Awareness = High, Attitude = High, Intention = High")
    print(f"  --> P(Behaviour = High | Awareness=H, Attitude=H, Intention=H) = {q1_high['Behaviour=High']:.4f} ({q1_high['Behaviour=High']*100:.2f}%)")
    print(f"  --> P(Behaviour = Low  | Awareness=H, Attitude=H, Intention=H) = {q1_high['Behaviour=Low']:.4f} ({q1_high['Behaviour=Low']*100:.2f}%)")
    print(f"  [Contrast] P(Behaviour = High | Awareness=H, Attitude=H, Intention=LOW) = {q1_low['Behaviour=High']:.4f} ({q1_low['Behaviour=High']*100:.2f}%)")
    print("  * Note: Due to d-separation in the DAG, Behaviour is conditionally independent of Awareness and Attitude given Intention.")

    # Required Query 2:
    # P(Intention = High | Trust = High, PBC = High, Social Influence = High)
    q2_high = run_query(['Intention'], {'Trust': 'High', 'PBC': 'High', 'Social_Influence': 'High'})
    q2_low  = run_query(['Intention'], {'Trust': 'Low', 'PBC': 'Low', 'Social_Influence': 'Low'})
    
    print("\n--- 3. Required Query 2: Purchase Intention Given Favourable Contextual Enablers ---")
    print("  Evidence: Trust = High, PBC = High, Social_Influence = High")
    print(f"  --> P(Intention = High | Trust=H, PBC=H, Social_Influence=H) = {q2_high['Intention=High']:.4f} ({q2_high['Intention=High']*100:.2f}%)")
    print(f"  --> P(Intention = Low  | Trust=H, PBC=H, Social_Influence=H) = {q2_high['Intention=Low']:.4f} ({q2_high['Intention=Low']*100:.2f}%)")
    print(f"  [Contrast] P(Intention = High | Trust=L, PBC=L, Social_Influence=L) = {q2_low['Intention=High']:.4f} ({q2_low['Intention=High']*100:.2f}%)")

    # Additional Scenario 3: Price Barrier Impact
    q3_barrier_high = run_query(['Intention'], {'Price_Sensitivity': 'High', 'Attitude': 'High', 'Trust': 'High'})
    q3_barrier_low  = run_query(['Intention'], {'Price_Sensitivity': 'Low',  'Attitude': 'High', 'Trust': 'High'})
    print("\n--- 4. Scenario 3: Economic Barrier Effect on Purchase Intention ---")
    print(f"  --> P(Intention = High | Price_Barrier = High, Attitude = High, Trust = High) = {q3_barrier_high['Intention=High']:.4f} ({q3_barrier_high['Intention=High']*100:.2f}%)")
    print(f"  --> P(Intention = High | Price_Barrier = Low,  Attitude = High, Trust = High) = {q3_barrier_low['Intention=High']:.4f} ({q3_barrier_low['Intention=High']*100:.2f}%)")

    # Additional Scenario 4: Diagnostic (Abductive) Reasoning
    q4_diag_att_h = run_query(['Attitude'], {'Behaviour': 'High'})
    q4_diag_att_l = run_query(['Attitude'], {'Behaviour': 'Low'})
    q4_diag_awa_h = run_query(['Awareness'], {'Behaviour': 'High'})
    q4_diag_awa_l = run_query(['Awareness'], {'Behaviour': 'Low'})

    print("\n--- 5. Scenario 4: Diagnostic (Abductive / Backward) Reasoning ---")
    print(f"  --> P(Attitude  = High | Behaviour = High) = {q4_diag_att_h['Attitude=High']:.4f} vs P(Attitude = High | Behaviour = Low) = {q4_diag_att_l['Attitude=High']:.4f}")
    print(f"  --> P(Awareness = High | Behaviour = High) = {q4_diag_awa_h['Awareness=High']:.4f} vs P(Awareness = High | Behaviour = Low) = {q4_diag_awa_l['Awareness=High']:.4f}")

    # ---------------------------------------------------------
    # 9. GENERATE PROFESSIONAL PLOTS
    # ---------------------------------------------------------
    print("\n" + "="*85)
    print(" GENERATING VISUAL PLOTS ")
    print("="*85)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#ffffff')

    # Chart 1: Key Queries Comparison
    query_names = [
        'P(Behaviour=High |\nAware=H, Att=H, Int=H)',
        'P(Behaviour=High |\nAware=H, Att=H, Int=LOW)',
        'P(Intention=High |\nTrust=H, PBC=H, Soc=H)',
        'P(Intention=High |\nTrust=L, PBC=L, Soc=L)',
        'P(Intention=High |\nBarrier=LOW, Att=H, Trust=H)',
        'P(Intention=High |\nBarrier=HIGH, Att=H, Trust=H)'
    ]
    query_probs = [
        q1_high['Behaviour=High'],
        q1_low['Behaviour=High'],
        q2_high['Intention=High'],
        q2_low['Intention=High'],
        q3_barrier_low['Intention=High'],
        q3_barrier_high['Intention=High']
    ]
    bar_colors = ['#1b5e20', '#b71c1c', '#0d47a1', '#e65100', '#00695c', '#880e4f']

    y_pos = np.arange(len(query_names))
    axes[0].barh(y_pos, query_probs, color=bar_colors, height=0.55, edgecolor='black', alpha=0.9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(query_names, fontsize=9.5, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel('Posterior Probability', fontsize=11, fontweight='bold')
    axes[0].set_title('Bayesian Inference Query Results', fontsize=12, fontweight='bold', pad=10)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    for i, v in enumerate(query_probs):
        axes[0].text(v + 0.02, i, f"{v:.1%}", va='center', fontweight='bold', fontsize=9.5)

    # Chart 2: Prior Node Marginals
    node_labels = ['Awareness', 'Attitude', 'Social Influ.', 'PBC', 'Price Barrier', 'Trust', 'Intention', 'Behaviour']
    node_vals = [marginals[n] for n in nodes]

    axes[1].bar(node_labels, node_vals, color='#283593', width=0.52, edgecolor='black', alpha=0.85)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('Probability P(State = High)', fontsize=11, fontweight='bold')
    axes[1].set_title('Prior Marginal Probabilities P(Node = High)', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xticks(range(len(node_labels)))
    axes[1].set_xticklabels(node_labels, rotation=35, ha='right', fontsize=9.5, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for i, v in enumerate(node_vals):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plot_file = 'bayesian_network_charts.png'
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved visualization plot to: {plot_file}")

    # Export inference results table to CSV
    results_df = pd.DataFrame([
        {'Query': 'P(Behaviour=High | Awareness=High, Attitude=High, Intention=High)', 'P_High': q1_high['Behaviour=High'], 'P_Low': q1_high['Behaviour=Low']},
        {'Query': 'P(Behaviour=High | Awareness=High, Attitude=High, Intention=Low)', 'P_High': q1_low['Behaviour=High'], 'P_Low': q1_low['Behaviour=Low']},
        {'Query': 'P(Intention=High | Trust=High, PBC=High, Social_Influence=High)', 'P_High': q2_high['Intention=High'], 'P_Low': q2_high['Intention=Low']},
        {'Query': 'P(Intention=High | Trust=Low, PBC=Low, Social_Influence=Low)', 'P_High': q2_low['Intention=High'], 'P_Low': q2_low['Intention=Low']},
        {'Query': 'P(Intention=High | Price_Barrier=Low, Attitude=High, Trust=High)', 'P_High': q3_barrier_low['Intention=High'], 'P_Low': q3_barrier_low['Intention=Low']},
        {'Query': 'P(Intention=High | Price_Barrier=High, Attitude=High, Trust=High)', 'P_High': q3_barrier_high['Intention=High'], 'P_Low': q3_barrier_high['Intention=Low']},
        {'Query': 'P(Attitude=High | Behaviour=High)', 'P_High': q4_diag_att_h['Attitude=High'], 'P_Low': q4_diag_att_h['Attitude=Low']},
        {'Query': 'P(Attitude=High | Behaviour=Low)', 'P_High': q4_diag_att_l['Attitude=High'], 'P_Low': q4_diag_att_l['Attitude=Low']},
        {'Query': 'P(Awareness=High | Behaviour=High)', 'P_High': q4_diag_awa_h['Awareness=High'], 'P_Low': q4_diag_awa_h['Awareness=Low']},
        {'Query': 'P(Awareness=High | Behaviour=Low)', 'P_High': q4_diag_awa_l['Awareness=High'], 'P_Low': q4_diag_awa_l['Awareness=Low']}
    ])
    results_df.to_csv('bayesian_inference_results.csv', index=False)
    print("Saved inference results table to: bayesian_inference_results.csv")
    print("\nAll Bayesian Network computations completed successfully!")

if __name__ == '__main__':
    main()
