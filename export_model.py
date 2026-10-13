import json
import itertools
import pandas as pd
import numpy as np

# Load CSV
csv_file = "PGM GOOGLE FORM RESPONSES.csv"
df_raw = pd.read_csv(csv_file)

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

# Convert numeric Likert items
likert_cols = ['Awareness', 'Attitude', 'Social_Influence', 'PBC', 
               'Price_Sensitivity', 'Trust', 'Intention', 
               'Sustainable_Preference', 'Willingness_To_Pay']
for c in likert_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Economic barrier index: (Price_Sensitivity + (6 - WTP)) / 2
df['WTP_Barrier'] = 6 - df['Willingness_To_Pay']
df['Economic_Barrier_Index'] = (df['Price_Sensitivity'] + df['WTP_Barrier']) / 2.0

# Discretization
df_bn = pd.DataFrame()
df_bn['Awareness'] = np.where(df['Awareness'] >= 4, 'High', 'Low')
df_bn['Attitude'] = np.where(df['Attitude'] >= 4, 'High', 'Low')
df_bn['Social_Influence'] = np.where(df['Social_Influence'] >= 4, 'High', 'Low')
df_bn['PBC'] = np.where(df['PBC'] >= 4, 'High', 'Low')
df_bn['Price_Sensitivity'] = np.where(df['Economic_Barrier_Index'] >= 3.5, 'High', 'Low')
df_bn['Trust'] = np.where(df['Trust'] >= 4, 'High', 'Low')
df_bn['Intention'] = np.where(df['Intention'] >= 4, 'High', 'Low')

# Handle Behaviour properly:
# 'None' or NaN when 'None' read by pandas -> 'Low' (0 purchases)
# '1-2 items', '3-5 items', '6+ items' -> 'High' (active sustainable buyer)
beh_raw_str = df['Behaviour_Raw'].fillna('None').astype(str).str.strip().str.lower()
df_bn['Behaviour'] = np.where(beh_raw_str.str.contains('none'), 'Low', 'High')

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

alpha = 1.0  # Laplace smoothing

cpts = {}
cpt_export = {}

for node in nodes:
    par_list = parents[node]
    if not par_list:
        counts = {s: int((df_bn[node] == s).sum()) for s in state_names}
        total = len(df_bn) + 2 * alpha
        probs = {s: float((counts[s] + alpha) / total) for s in state_names}
        cpts[node] = probs
        cpt_export[node] = {
            'type': 'prior',
            'parents': [],
            'counts': counts,
            'probs': probs
        }
    else:
        par_configs = list(itertools.product(state_names, repeat=len(par_list)))
        cpt_table = {}
        export_table = {}
        for cfg in par_configs:
            cond = pd.Series(True, index=df_bn.index)
            for p_name, p_val in zip(par_list, cfg):
                cond = cond & (df_bn[p_name] == p_val)
            n_matching = int(cond.sum())
            counts = {s: int((cond & (df_bn[node] == s)).sum()) for s in state_names}
            total = n_matching + 2 * alpha
            probs = {s: float((counts[s] + alpha) / total) for s in state_names}
            cpt_table[cfg] = probs
            key_str = "|".join(f"{p}={v}" for p, v in zip(par_list, cfg))
            export_table[key_str] = {
                'counts': counts,
                'total_matching': n_matching,
                'probs': probs
            }
        cpts[node] = cpt_table
        cpt_export[node] = {
            'type': 'conditional',
            'parents': par_list,
            'table': export_table
        }

# Also export all 22 respondents for transparency
respondents = []
for idx, row in df_bn.iterrows():
    respondents.append(row.to_dict())

export_data = {
    'nodes': nodes,
    'parents': parents,
    'states': state_names,
    'cpts': cpt_export,
    'sample_size': len(df_bn),
    'respondents': respondents,
    'marginal_counts': {col: df_bn[col].value_counts().to_dict() for col in nodes}
}

with open('model_data.json', 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2)

print("Exported model_data.json successfully!")
print("Behaviour counts:", df_bn['Behaviour'].value_counts().to_dict())
