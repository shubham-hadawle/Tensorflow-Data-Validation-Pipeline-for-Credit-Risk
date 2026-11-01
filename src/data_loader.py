import os
import numpy as np
import pandas as pd

def generate_credit_risk_datasets(output_dir="data/raw", n_train=10000, n_eval=2500, n_serving=2500, n_drifted=3000, seed=42):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    loan_intents = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    home_ownerships = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    default_history = ['N', 'Y']

    def generate_base(n, is_drift=False):
        if not is_drift:
            age = np.random.randint(21, 65, size=n)
            inc = np.random.lognormal(mean=10.8, sigma=0.55, size=n)
            score = np.clip(np.random.normal(loc=680, scale=60, size=n).astype(int), 350, 850)
            loan = np.clip(np.random.gamma(shape=3.5, scale=4000, size=n), 1000, 45000)
            dti = np.clip(np.random.beta(a=2.5, b=6.0, size=n) * 0.7, 0.05, 0.65)
            emp = np.clip(np.random.exponential(scale=5.0, size=n), 0.0, 35.0)
            intent = np.random.choice(loan_intents, size=n, p=[0.20, 0.18, 0.15, 0.12, 0.15, 0.20])
            home = np.random.choice(home_ownerships, size=n, p=[0.48, 0.40, 0.10, 0.02])
            tenure = np.random.randint(6, 180, size=n)
            ext_risk = np.clip(np.random.beta(2, 5, size=n), 0.02, 0.98)
            hist = np.random.choice(default_history, size=n, p=[0.85, 0.15])
        else:
            age = np.random.randint(20, 68, size=n)
            inc = np.random.lognormal(mean=10.4, sigma=0.60, size=n)
            score = np.clip(np.random.normal(loc=630, scale=70, size=n).astype(int), 350, 850)
            loan = np.clip(np.random.gamma(shape=4.0, scale=4500, size=n), 1000, 50000)
            dti = np.clip(np.random.beta(a=3.5, b=4.5, size=n) * 0.8, 0.08, 0.75)
            emp = np.clip(np.random.exponential(scale=3.8, size=n), 0.0, 30.0)
            intent = np.random.choice(loan_intents, size=n, p=[0.08, 0.08, 0.12, 0.06, 0.16, 0.50])
            home = np.random.choice(home_ownerships, size=n, p=[0.65, 0.25, 0.08, 0.02])
            tenure = np.random.randint(3, 160, size=n)
            ext_risk = np.clip(np.random.beta(3, 3, size=n), 0.05, 0.99)
            hist = np.random.choice(default_history, size=n, p=[0.72, 0.28])

        log_odds = (-2.5 + 3.2 * dti - 0.005 * (score - 600) - 0.000015 * inc + 0.000035 * loan + 1.8 * ext_risk + (hist == 'Y') * 1.4 + (intent == 'DEBTCONSOLIDATION') * 0.4 + (home == 'RENT') * 0.3)
        prob = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -10, 10)))
        is_def = (np.random.rand(n) < prob).astype(int)

        return pd.DataFrame({
            'customer_age': age, 'annual_income': np.round(inc, 2), 'credit_score': score,
            'loan_amount': np.round(loan, 2), 'loan_intent': intent, 'employment_length_years': np.round(emp, 1),
            'debt_to_income_ratio': np.round(dti, 4), 'home_ownership': home, 'account_tenure_months': tenure,
            'historical_default': hist, 'risk_score_external': np.round(ext_risk, 4), 'is_default': is_def
        })

    train_df = generate_base(n_train, False)
    train_df.loc[np.random.rand(n_train) < 0.03, 'employment_length_years'] = np.nan

    eval_df = generate_base(n_eval, False)
    anom_idx = np.random.choice(n_eval, size=25, replace=False)
    eval_df.loc[anom_idx[:15], 'loan_intent'] = 'CRYPTO_MINING'
    eval_df.loc[anom_idx[15:], 'loan_intent'] = 'TAX_SETTLEMENT'
    eval_df.loc[np.random.rand(n_eval) < 0.28, 'employment_length_years'] = np.nan
    eval_df.loc[np.random.choice(n_eval, size=5, replace=False), 'debt_to_income_ratio'] = -0.15

    serving_df = generate_base(n_serving, False).drop(columns=['is_default'])
    serving_df.loc[np.random.rand(n_serving) < 0.04, 'employment_length_years'] = np.nan

    drifted_df = generate_base(n_drifted, True)
    drifted_df.loc[np.random.rand(n_drifted) < 0.05, 'employment_length_years'] = np.nan

    train_df.to_csv(os.path.join(output_dir, "train_credit_risk.csv"), index=False)
    eval_df.to_csv(os.path.join(output_dir, "eval_credit_risk.csv"), index=False)
    serving_df.to_csv(os.path.join(output_dir, "serving_credit_risk.csv"), index=False)
    drifted_df.to_csv(os.path.join(output_dir, "drifted_batch_credit_risk.csv"), index=False)
    return train_df, eval_df, serving_df, drifted_df

def load_datasets(data_dir="data/raw"):
    p_train = os.path.join(data_dir, "train_credit_risk.csv")
    if not os.path.exists(p_train):
        return generate_credit_risk_datasets(output_dir=data_dir)
    return (pd.read_csv(p_train), pd.read_csv(os.path.join(data_dir, "eval_credit_risk.csv")),
            pd.read_csv(os.path.join(data_dir, "serving_credit_risk.csv")),
            pd.read_csv(os.path.join(data_dir, "drifted_batch_credit_risk.csv")))
