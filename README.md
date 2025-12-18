# Fintech Project: Credit Card Default Prediction

## 📌 Project Overview
This project focuses on a critical financial analytics problem: **predicting credit card default**. Using the "Default of Credit Card Clients" dataset, we build and compare machine learning models to identify customers at risk of defaulting on their payments.

The project demonstrates an end-to-end data science workflow:
1.  **Data Ingestion**: Loading data from OpenML.
2.  **Preprocessing**: Cleaning, renaming, and scaling features.
3.  **EDA**: Visualizing distributions, correlations, and risk factors.
4.  **Modeling**: Comparing **Logistic Regression** (Baseline) vs. **XGBoost** (Advanced).
5.  **Reporting**: Automated generation of a PDF summary report.

## 📂 Project Structure
```
Fintech_Loan_Default_Prediction/
├── images/                         # Generated plots for the report
├── Loan_Default_Prediction.ipynb   # Interactive Jupyter Notebook (Visual Analysis)
├── Loan_Default_Prediction_Report.pdf # Final PDF Report Artifact
├── requirements.txt                # Python dependencies
├── run_analysis.py                 # Standalone script to run analysis & generate PDF
└── README.md                       # Project documentation
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Installation
1.  Clone the repository or download the folder.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### Option 1: Jupyter Notebook (Interactive)
Open the notebook to explore the data and run cells step-by-step:
```bash
jupyter notebook Loan_Default_Prediction.ipynb
```

### Option 2: Python Script (Automated)
Run the script to execute the full pipeline and generate the PDF report:
```bash
python run_analysis.py
```
*   This will save plots to the `images/` folder.
*   It will generate `Loan_Default_Prediction_Report.pdf`.

## 📊 Key Results
- **Key Predictors**: The most recent repayment status (`PAY_1`) and the credit limit (`LIMIT_BAL`) were identified as the strongest indicators of default risk.
- **Model Performance**:
    - **Logistic Regression AUC**: ~0.71
    - **XGBoost AUC**: ~0.76
    - *Conclusion*: The XGBoost model successfully captured non-linear risk patterns and outperformed the linear baseline.

## 📝 License
This project is for educational purposes. The dataset is sourced from the UCI Machine Learning Repository via OpenML.
