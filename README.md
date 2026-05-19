# Telco Customer Churn Prediction & Business Optimization

## 📌 Project Overview
Customer churn is one of the most significant challenges in the telecommunications industry. Retaining an existing customer is highly cost-effective compared to acquiring a new one. This project provides a robust, end-to-end machine learning solution to predict customer churn. Beyond just building a predictive model, this project bridges the gap between data science and business strategy by utilizing Lift Curve analysis, precise threshold tuning, and cost-benefit evaluation to maximize profitability.

## 📊Business Value & Strategy
The core of this project is transforming raw machine learning metrics into actionable business strategies. We focused on the financial impact of customer retention campaigns.

* **Lift Curve Analysis:** By targeting customers based on their predicted risk probabilities, the model successfully identifies the highest-risk segment, ensuring that marketing budgets are spent on the right audience.
* **Cost Optimization:** Assuming a $500 cost constraint per lost customer, the decision threshold was adjusted to minimize financial loss.
* **Threshold Tuning:** The optimal classification threshold was lowered to `0.23`. This aggressive strategy yielded a Recall of 94%, meaning the model successfully identifies 94% of all churning customers, ensuring virtually no high-risk customer slips through unnoticed.
* **Hybrid Machine Learning** (Unsupervised + Supervised): Engineered a highly predictive feature (Dist_to_Cluster) by first clustering customers (Unsupervised) and passing their cluster distances as input features to the classification model (Supervised).
* **Explainable AI (XAI)**: Utilized SHAP (Waterfall and Summary plots) to break down model predictions, enabling stakeholders to understand exactly why a specific customer is at risk of churning.

## Machine Learning Approach & Methodology
To ensure the model is reliable and ready for production, strict data science best practices were implemented:

* **Leak-Proof Pipeline:** Data preprocessing, imputation, and scaling were strictly integrated within a Scikit-Learn `Pipeline`. This guarantees zero data leakage during the training and validation phases.
* **Robust Cross-Validation:** A 5-fold Cross-Validation (K-Fold = 5) strategy was applied. The results demonstrated extremely low variance (CV Std = 0.0112) and highly consistent scores between the training folds and the unseen test set (CV Mean: 0.8494 vs. Test AUC: 0.8482). This confirms the model is highly stable and free from overfitting.
* **Metric Focus:** The model was evaluated primarily using ROC-AUC for overall ranking capability and Recall for business-specific sensitivity.

## ⭐Key Results
* **Cross-Validation ROC-AUC:** 0.8494 (± 0.0112)
* **Test ROC-AUC:** 0.8482
* **Recall at optimal threshold (0.23):** 0.94
* **Stability:** Perfect generalization from training folds to unseen test data.

## 🛠️Technologies Used
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
* **Techniques:** Feature Engineering, Pipeline Architecture,SHAP Cross-Validation, Lift/Gain Charts, Probability Calibration,

## 📈 Results & Visualizations
* **Lift Curve**: Demonstrated a Lift of ~3.5x for the top percentiles, meaning the model is 3.5 times better at identifying churners than random guessing.
* **SHAP Summary Plot**: Revealed that factors like Tenure, Monthly Charges, and Contract Type are the most influential drivers of customer churn.
* **Business Impact**: By shifting the threshold from the default 0.5 to the financially optimized value, the system successfully captures a significantly higher percentage of actual churners, preventing massive revenue leakage.

## How to Run

1. Clone this repository:
git clone https://github.com/your-username/Telco-Churn-Optimization.git

2. Install the required dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn

3. Run the Jupyter Notebook or Python scripts provided in the repository to view the data pipeline, model training, and Lift Curve visualizations.
