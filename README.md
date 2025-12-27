# 🚗 Car Insurance Claim Prediction & Risk Analysis

An end-to-end **Machine Learning project** that predicts the probability of an insurance claim and visualizes insights using an interactive **Power BI dashboard**.
The project focuses on **model performance, explainability, and business-driven risk segmentation**.

---

## 📌 Project Overview

Insurance companies need to accurately assess the likelihood of claims to:
- Reduce financial risk
- Improve underwriting decisions
- Optimize premium pricing

This project builds a **claim prediction model** using Python and translates model outputs into **actionable business insights** using Power BI.

## Dataset
Due to GitHub file size limits, large dataset ZIP files are excluded from this repository.

Please use the provided `train.csv` and `test.csv` files to run the project.
If the full dataset is required, it can be downloaded separately.


---

## 🧠 Key Features

- Data preprocessing & feature engineering
- Machine learning model training and evaluation
- Feature importance for model explainability
- Claim probability prediction
- Risk categorization (Low / Medium / High)
- Interactive Power BI dashboard for business users

---

## 🛠 Tech Stack

### Programming & Machine Learning
- Python 3.9
- Pandas, NumPy
- Scikit-learn
- Joblib

### Visualization & BI
- Power BI

### Tools
- VS Code
- Git
- GitHub

---

## 📂 Project Structure

CAR_INSURANCE_PROJECT/
│
├── model_training.py
├── preprocessing.py
├── predictions.py
├── feature_importance.py
├── evaluation.py
├── hyperparameter_tuning.py
│
├── train.csv
├── test.csv
├── predictions.csv
├── feature_importance.csv
│
├── car_insurance_powerbi.pbix
├── requirements.txt
└── README.md

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt

2️⃣ Train the model
python model_training.py

3️⃣ Generate predictions
python predictions.py

4️⃣ Extract feature importance
python feature_importance.py

