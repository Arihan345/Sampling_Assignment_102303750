# Sampling Assignment – Credit Card Fraud Detection

## Objective

The objective of this assignment is to analyze the impact of different **sampling techniques** on a **highly imbalanced dataset** and evaluate how these techniques affect the performance of various **machine learning models**.

This study is conducted on a real-world **credit card fraud detection dataset**, where fraudulent transactions are significantly fewer than legitimate ones.

---

## Project Structure

SAMPLING_ASSIGNMENT_102303750/
│
├── data/
│ └── Creditcard_data.csv
│
├── src/
│ ├── sampling.py
│ ├── models.py
│ └── main.py
│
├── results/
│ ├── accuracy_table.csv
│ ├── accuracy_heatmap.png
│ ├── original_class_distribution.png
│ └── smote_class_distribution.png
│
├── requirements.txt
└── README.md


---

## Dataset Description

- **Dataset:** Credit Card Fraud Dataset  
- **Target Column:** `Class`  
  - `0` → Normal Transaction  
  - `1` → Fraudulent Transaction  

The dataset is **highly imbalanced**, making sampling essential before training machine learning models.

---

## Sampling Techniques Used

The following five sampling techniques were applied using the `imbalanced-learn` library:

| Sampling Code | Sampling Technique |
|--------------|------------------|
| Sampling1 | Random Under Sampling |
| Sampling2 | Random Over Sampling |
| Sampling3 | SMOTE |
| Sampling4 | NearMiss |
| Sampling5 | SMOTEENN |

---

## Machine Learning Models Used

| Model Code | Model Name |
|----------|------------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | K-Nearest Neighbors (KNN) |
| M5 | Naive Bayes |

---

## Visualizations

All visualizations are automatically generated and saved in the `results/` directory.

### 📊 Original Class Distribution (Before Sampling)

This plot shows the severe class imbalance in the dataset.

![Original Class Distribution](results/original_class_distribution.png)

---

### 📊 Class Distribution After SMOTE Sampling

This plot confirms that SMOTE balances the dataset by generating synthetic minority samples.

![SMOTE Class Distribution](results/smote_class_distribution.png)

---

### 📊 Accuracy Comparison Heatmap

This heatmap compares the accuracy of each model across different sampling techniques.

![Accuracy Heatmap](results/accuracy_heatmap.png)

---

## Accuracy Results (From `accuracy_table.csv`)

The table below shows the **exact accuracy values (%)** obtained for each combination of sampling technique and model.

| Model | Sampling1<br>(Under) | Sampling2<br>(Over) | Sampling3<br>(SMOTE) | Sampling4<br>(NearMiss) | Sampling5<br>(SMOTEENN) |
|------|----------------------|---------------------|----------------------|--------------------------|--------------------------|
| **M1 (Logistic Regression)** | 66.67 | 91.92 | 90.39 | 50.00 | 96.24 |
| **M2 (Decision Tree)** | 66.67 | 98.25 | 98.25 | 16.67 | 98.84 |
| **M3 (Random Forest)** | 16.67 | 99.78 | 99.34 | 16.67 | 99.42 |
| **M4 (KNN)** | 16.67 | 98.47 | 84.72 | 83.33 | 94.22 |
| **M5 (Naive Bayes)** | 50.00 | 78.17 | 84.50 | 33.33 | 87.86 |

📁 Full results are available in:

results/accuracy_table.csv


---

## Observations and Discussion

- The original dataset was highly imbalanced, which negatively affected model learning.
- **Oversampling techniques (SMOTE, SMOTEENN)** significantly improved accuracy across most models.
- **Random Forest (M3)** consistently achieved the highest accuracy.
- **NearMiss** showed reduced performance due to aggressive undersampling.
- **SMOTEENN** provided the best overall balance between noise removal and oversampling.

---

## Conclusion

This experiment demonstrates that **sampling techniques play a critical role** in improving machine learning performance on imbalanced datasets.  
Appropriate sampling leads to better generalization, higher accuracy, and more reliable fraud detection systems.

---

## How to Run the Project

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/main.py
```