import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="FinWise AI", page_icon="💰", layout="wide")
st.title("💰 FinWise AI")
st.subheader("ML-Based Personal Financial Adviser")
st.caption("Educational project prototype. Estimates depend on the quality and suitability of the dataset.")

MODEL_PATH = "models/savings_model.joblib"
DATA_PATH = "data/personal_finance_tracker_dataset.csv"

if not os.path.exists(MODEL_PATH):
    st.warning("Model not trained yet. Run `python train_model.py` in the project folder.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
metrics = bundle.get("metrics", {})
model_name = bundle.get("model_name", "Saved model")

st.sidebar.header("Enter financial details")
st.sidebar.caption("Use the same units as your dataset.")
monthly_income = st.sidebar.number_input("Monthly income", min_value=0.0, value=4000.0, step=100.0)
monthly_expense_total = st.sidebar.number_input("Total monthly expenses", min_value=0.0, value=3000.0, step=100.0)
budget_goal = st.sidebar.number_input("Budget goal", min_value=0.0, value=2800.0, step=100.0)
credit_score = st.sidebar.number_input("Credit score", min_value=0.0, value=680.0, step=1.0)
debt_to_income_ratio = st.sidebar.slider("Debt-to-income ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
loan_payment = st.sidebar.number_input("Loan payment", min_value=0.0, value=500.0, step=50.0)
investment_amount = st.sidebar.number_input("Investment amount", min_value=0.0, value=400.0, step=50.0)
subscription_services = st.sidebar.number_input("Number of subscription services", min_value=0, value=5, step=1)
emergency_fund = st.sidebar.number_input("Emergency fund", min_value=0.0, value=1000.0, step=100.0)
transaction_count = st.sidebar.number_input("Transaction count", min_value=0, value=60, step=1)
discretionary_spending = st.sidebar.number_input("Discretionary spending", min_value=0.0, value=500.0, step=50.0)
essential_spending = st.sidebar.number_input("Essential spending", min_value=0.0, value=2200.0, step=50.0)
rent_or_mortgage = st.sidebar.number_input("Rent or mortgage", min_value=0.0, value=1200.0, step=50.0)

st.sidebar.subheader("Other details")
scenario_options = ["normal", "inflation", "recession"]
income_type_options = ["Salary", "Freelance", "Business"]
category_options = [
    "Investments", "Healthcare", "Groceries", "Utilities", "Transportation",
    "Entertainment", "Insurance", "Education", "Dining", "Other"
]
financial_scenario = st.sidebar.selectbox("Financial scenario", scenario_options)
income_type = st.sidebar.selectbox("Income type", income_type_options)
category = st.sidebar.selectbox("Main spending category", category_options)

input_row = {
    "monthly_income": monthly_income,
    "monthly_expense_total": monthly_expense_total,
    "budget_goal": budget_goal,
    "credit_score": credit_score,
    "debt_to_income_ratio": debt_to_income_ratio,
    "loan_payment": loan_payment,
    "investment_amount": investment_amount,
    "subscription_services": subscription_services,
    "emergency_fund": emergency_fund,
    "transaction_count": transaction_count,
    "discretionary_spending": discretionary_spending,
    "essential_spending": essential_spending,
    "rent_or_mortgage": rent_or_mortgage,
    "financial_scenario": financial_scenario,
    "income_type": income_type,
    "category": category,
}
input_df = pd.DataFrame([{c: input_row[c] for c in features}])
predicted_savings = max(0.0, float(model.predict(input_df)[0]))
balance = monthly_income - monthly_expense_total

m1, m2, m3 = st.columns(3)
m1.metric("Monthly income", f"{monthly_income:,.2f}")
m2.metric("Monthly expenses", f"{monthly_expense_total:,.2f}")
m3.metric("Income minus expenses", f"{balance:,.2f}")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Expense overview")
    chart_df = pd.DataFrame({
        "Type": ["Essential spending", "Discretionary spending", "Other expenses"],
        "Amount": [
            essential_spending,
            discretionary_spending,
            max(0.0, monthly_expense_total - essential_spending - discretionary_spending)
        ]
    })
    chart_df = chart_df[chart_df["Amount"] > 0]
    if not chart_df.empty:
        st.plotly_chart(px.pie(chart_df, names="Type", values="Amount",
                               title="Estimated expense composition"), use_container_width=True)
    else:
        st.info("Enter expense values to see the chart.")

with right:
    st.subheader("Savings prediction")
    st.metric("Predicted actual savings", f"{predicted_savings:,.2f}")
    st.caption(f"Model used: {model_name}")
    st.caption("This predicts the dataset's `actual_savings` field, not a guaranteed next-month amount.")

st.subheader("Budgeting suggestions")
suggestions = []
if balance < 0:
    suggestions.append("Your listed expenses exceed income. Review your records and prioritize essential obligations.")
else:
    suggestions.append(f"Income minus expenses is {balance:,.2f}. Consider setting a realistic savings target after essential costs.")
if monthly_income > 0 and rent_or_mortgage / monthly_income > 0.35:
    suggestions.append("Rent or mortgage is above 35% of income in these inputs; review whether this is sustainable.")
if monthly_income > 0 and loan_payment / monthly_income > 0.25:
    suggestions.append("Loan payments exceed 25% of income in these inputs. Review obligations carefully.")
if monthly_income > 0 and discretionary_spending / monthly_income > 0.20:
    suggestions.append("Review discretionary purchases to see whether any spending can be redirected toward your goals.")
for s in suggestions:
    st.write("• " + s)

st.subheader("Savings goal calculator")
goal = st.number_input("Savings goal", min_value=0.0, value=10000.0, step=500.0)
monthly_contribution = st.number_input("Planned monthly contribution", min_value=0.0, value=1000.0, step=100.0)
if goal == 0:
    st.info("Enter a positive savings goal.")
elif monthly_contribution > 0:
    months = int((goal + monthly_contribution - 1) // monthly_contribution)
    st.success(f"At a constant contribution, the goal would take about {months} months (ignoring interest and unexpected expenses).")
else:
    st.info("Enter a monthly contribution greater than zero.")

with st.expander("Model evaluation"):
    st.write("Metrics from the held-out later-date portion of the provided dataset:")
    if metrics:
        rows = [{"Model": name, "MAE": vals["mae"], "R²": vals["r2"]} for name, vals in metrics.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("These metrics measure performance on this dataset only. They do not guarantee performance for real users.")
with st.expander("Dataset notes"):
    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        st.write(f"Dataset loaded: {len(data)} rows and {len(data.columns)} columns.")
        st.write("The model excludes IDs, dates, and several derived/outcome columns to reduce leakage.")
    st.write("Confirm the dataset's source, definitions, units, and suitability before presenting predictions as real-world financial guidance.")
