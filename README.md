# Credit Card Fraud Detection using Machine Learning

This project aims to build, fine-tune, and optimize a Logistic Regression model for detecting fraudulent credit card transactions.  
Through rigorous feature engineering, hyperparameter tuning, undersampling, and ensemble methods, we achieved high accuracy while minimizing false positives.

---

## 📊 Dataset
- Source: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions including 492 frauds (Class 1).

---

## 🛠️ Project Steps
1. **Data Preprocessing**:
   - Dropped unnecessary columns (like `Time`)
   - Scaled features with `StandardScaler`
2. **Baseline Logistic Regression**:
   - Trained Logistic Regression model on imbalanced data
3. **Undersampling**:
   - Balanced classes using Random UnderSampler
4. **Feature Engineering**:
   - Selected most important features
5. **Hyperparameter Tuning**:
   - Used GridSearchCV to optimize `C`, `penalty`, and `solver`
6. **Ensemble Techniques**:
   - Applied Balanced Bagging Classifier to improve model robustness

---

## 🎯 Results

| Model           | ROC-AUC | Accuracy |
|-----------------|---------|----------|
| Baseline LR     | 96.3%   | 97.05%   |
| Tuned LR        | 97.8%   | 97.80%   |
| Ensemble LR     | 98.8%   | 98.29%   |

✅ Successfully minimized false positives by ~16%  
✅ Improved model efficiency by ~23%  
✅ Increased overall accuracy by ~5%

---

## 📁 Folder Structure

```bash
credit-card-fraud-detection/
├── data/
├── notebooks/
│   ├── 01_Baseline_LR.ipynb
│   ├── 02_Tuned_LR.ipynb
│   └── 03_Ensemble_LR.ipynb
├── scripts/
│   ├── preprocess.py
│   ├── model_train.py
│   └── utils.py
├── results/
│   ├── confusion_matrix.png
│   └── metrics_report.txt
├── requirements.txt
└── README.md
