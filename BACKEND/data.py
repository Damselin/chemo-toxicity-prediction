import pandas as pd
import numpy as np

def generate_data(n=1000):
    np.random.seed(42)

    df = pd.DataFrame({
        'age': np.random.randint(25, 80, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'bmi': np.random.normal(25, 4, n),
        'cycle': np.random.randint(1, 6, n),
        'creatinine': np.random.normal(1.0, 0.3, n),
        'anc': np.random.normal(4.0, 1.2, n),
        'rash': np.random.randint(0, 4, n),
        'neuropathy': np.random.randint(0, 4, n),
        'nausea': np.random.randint(0, 4, n),
        'fatigue': np.random.randint(0, 4, n)
    })

    toxicity_prob = (
        0.05 * df['age'] +
        0.1 * df['rash'] +
        0.1 * df['neuropathy'] +
        0.1 * df['nausea'] +
        0.15 * (df['creatinine'] > 1.3).astype(int) +
        np.random.normal(0, 0.05, n)
    )

    df['toxicity'] = (toxicity_prob > np.percentile(toxicity_prob, 75)).astype(int)

    return df