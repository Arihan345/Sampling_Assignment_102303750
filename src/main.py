import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from sampling import load_data, get_samples
from models import get_models

os.makedirs("results", exist_ok=True)

X, y = load_data()
samples = get_samples(X, y)
models = get_models()

def plot_class_distribution(y, title, filename):
    counts = y.value_counts()

    plt.figure()
    counts.plot(kind='bar')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()

plot_class_distribution(
    y,
    "Original Class Distribution",
    "original_class_distribution.png"
)

X_smote, y_smote = samples["Sampling3"]
plot_class_distribution(
    y_smote,
    "Class Distribution After SMOTE",
    "smote_class_distribution.png"
)

results = []

for s_name, (X_s, y_s) in samples.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=0.3, random_state=42
    )

    for m_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100

        results.append({
            "Model": m_name,
            "Sampling": s_name,
            "Accuracy": round(acc, 2)
        })

df_results = pd.DataFrame(results)
df_results.to_csv("results/accuracy_table.csv", index=False)

print(df_results)

pivot_table = df_results.pivot(
    index="Model",
    columns="Sampling",
    values="Accuracy"
)

plt.figure(figsize=(8, 5))
sns.heatmap(pivot_table, annot=True, fmt=".2f")
plt.title("Model vs Sampling Accuracy Heatmap")
plt.tight_layout()
plt.savefig("results/accuracy_heatmap.png")
plt.close()
