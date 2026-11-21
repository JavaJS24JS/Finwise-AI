# FinWise AI — adapted to your uploaded dataset

## What this project does
- Trains Linear Regression and Random Forest Regressor.
- Predicts the dataset's `actual_savings` column.
- Uses numerical and categorical features with imputation and one-hot encoding.
- Splits by date: earlier dates for training, later dates for testing.
- Compares models using MAE and R² and saves the lower-MAE model.
- Provides a Streamlit dashboard with inputs, charts, suggestions, and goal calculator.

## Run on Windows / VS Code

Open the terminal inside this folder and run:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The dataset is already included in `data/personal_finance_tracker_dataset.csv`.

## Feature choices
Target: `actual_savings`

Features:
- Numeric: monthly_income, monthly_expense_total, budget_goal, credit_score, debt_to_income_ratio, loan_payment, investment_amount, subscription_services, emergency_fund, transaction_count, discretionary_spending, essential_spending, rent_or_mortgage
- Categorical: financial_scenario, income_type, category

Excluded to reduce leakage or because they are identifiers/derived labels: date (used only for splitting), user_id, savings_rate, cash_flow_status, financial_advice_score, financial_stress_level, savings_goal_met, fraud_flag.

## Important limitations
- The target is `actual_savings`, so the model predicts that field from other columns in the same row. This is not automatically a true next-month forecast.
- The dataset contains repeated user/date combinations and unevenly spaced dates. This starter splits by date but does not claim a rigorous person-level or time-series evaluation.
- Confirm the dataset's provenance and the meaning of each column. Check the train/test results and do not claim the model is accurate in the real world just because it runs.
- If the project requirement specifically says next-month savings, the dataset needs a reliable way to connect a person's current month to the following month's actual savings; the present file's irregular user records make that more difficult.

## Files
- `train_model.py`: preprocessing, training, comparison, model saving
- `app.py`: dashboard
- `data/personal_finance_tracker_dataset.csv`: provided dataset
- `models/`: saved model is created here
