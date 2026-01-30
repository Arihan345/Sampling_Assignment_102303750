# Sampling Assignment – Credit Card Fraud Detection

## Objective

The objective of this assignment is to understand the importance of **sampling techniques** in handling **highly imbalanced datasets** and to analyze how different sampling strategies affect the performance of various **machine learning models**.

In real-world applications such as credit card fraud detection, class imbalance can severely bias model predictions. This assignment demonstrates how appropriate sampling techniques improve model performance.

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

- **Dataset Name:** Creditcard_data.csv
- **Domain:** Credit Card Fraud Detection
- **Target Column:** `Class`
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction

The dataset is **highly imbalanced**, with fraudulent transactions forming a very small percentage of the data.

---

## Sampling Techniques Used

The imbalanced dataset was converted into balanced datasets using the following **five sampling techniques** from the `imbalanced-learn` library:

| Sampling Code | Sampling Technique | Description |
|--------------|------------------|------------|
| Sampling1 | Random Under Sampling | Reduces majority class samples to balance the dataset |
| Sampling2 | Random Over Sampling | Duplicates minority class samples randomly |
| Sampling3 | SMOTE | Generates synthetic minority class samples |
| Sampling4 | NearMiss | Undersampling based on nearest neighbors |
| Sampling5 | SMOTEENN | Combination of SMOTE and noise removal |

Each technique produces a balanced dataset suitable for training machine learning models.

---

## Machine Learning Models Used

Five different machine learning models were trained on each sampled dataset:

| Model Code | Model Name |
|----------|------------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | K-Nearest Neighbors (KNN) |
| M5 | Naive Bayes |

---

## Experimental Methodology

1. Load the original imbalanced dataset.
2. Apply each sampling technique to balance the dataset.
3. Split the sampled data into **70% training** and **30% testing** sets.
4. Train all five machine learning models on each sampled dataset.
5. Evaluate performance using **Accuracy**.
6. Save results and generate visualizations.

---

## Visualizations (Explicit Requirements)

To clearly demonstrate the effect of sampling techniques, the following visualizations were generated and saved automatically in the `results/` directory.

---

### 📊 Visualization 1: Original Class Distribution

**File:**

results/original_class_distribution.png


**Purpose:**
- Demonstrates the severe class imbalance in the original dataset.
- Justifies the need for applying sampling techniques.

**Description:**
This bar chart shows the distribution of normal and fraudulent transactions before applying any sampling.

---

### 📊 Visualization 2: Class Distribution After Sampling (SMOTE)

**File:**

results/smote_class_distribution.png


**Purpose:**
- Confirms that sampling successfully balances the dataset.
- Shows near-equal representation of both classes.

**Description:**
This plot shows the class distribution after applying **SMOTE**, which creates synthetic minority samples.

---

### 📊 Visualization 3: Accuracy Comparison Heatmap

**File:**

results/accuracy_heatmap.png


**Purpose:**
- Compares the accuracy of different machine learning models across sampling techniques.
- Helps identify the best sampling method for each model.

**Description:**
The heatmap displays accuracy values for all combinations of sampling techniques and models.

---

## Results

- All accuracy values are stored in:

results/accuracy_table.csv


- The table contains:
  - Sampling technique
  - Machine learning model
  - Accuracy percentage

---

## Observations and Discussion

- The original dataset was highly imbalanced, leading to biased model predictions.
- Oversampling techniques such as **SMOTE** and **SMOTEENN** significantly improved accuracy.
- **Random Forest** consistently achieved the highest performance across most sampling techniques.
- **NearMiss** resulted in lower accuracy due to aggressive removal of majority class samples.
- No single sampling technique is optimal for all models.

---

## Conclusion

This assignment demonstrates that **sampling techniques play a critical role** in improving machine learning performance on imbalanced datasets.  
Appropriate sampling leads to better generalization, fairer classification, and more reliable fraud detection systems.

---

## How to Run the Project

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the program
python3 src/main.py
