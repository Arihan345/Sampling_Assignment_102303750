import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN

def load_data():
    df = pd.read_csv("data/Creditcard_data.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def get_samples(X, y):
    samplers = {
        "Sampling1": RandomUnderSampler(random_state=42),
        "Sampling2": RandomOverSampler(random_state=42),
        "Sampling3": SMOTE(random_state=42),
        "Sampling4": NearMiss(),
        "Sampling5": SMOTEENN(random_state=42)
    }

    sampled_data = {}

    for name, sampler in samplers.items():
        X_res, y_res = sampler.fit_resample(X, y)
        sampled_data[name] = (X_res, y_res)

    return sampled_data
