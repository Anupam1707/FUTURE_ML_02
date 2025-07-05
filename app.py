import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Churn Predictor", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_churn = load_lottieurl(lottie_url)

st.markdown("<h1 style='color:#00c3ff;'>💼 Bank Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("Use machine learning to predict customer churn risk", unsafe_allow_html=True)

st_lottie(lottie_churn, height=180, key="churn")

model = joblib.load("model.pkl")
df = pd.read_csv("Bank Customer Churn Prediction.csv")
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={col: col.replace(" ", "") for col in df.columns}, inplace=True)

with st.form("churn_form"):
    st.subheader("🔧 Customer Information")

    credit_score = st.slider("Credit Score", 300, 900, 650)
    age = st.slider("Age", 18, 100, 40)
    balance = st.number_input("Balance ($)", 0.0, 300000.0, 50000.0)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    is_active = st.radio("Active Member?", ["Yes", "No"])
    has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
    gender = st.radio("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ["France", "Germany", "Spain"])
    estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 300000.0, 50000.0)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    gender = 1 if gender == "Male" else 0
    is_active = 1 if is_active == "Yes" else 0
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    country_ge = 1 if country == "Germany" else 0
    country_sp = 1 if country == "Spain" else 0

    input_df = pd.DataFrame([{
        'credit_score': credit_score,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': num_products,
        'credit_card': has_cr_card,
        'active_member': is_active,
        'estimated_salary': estimated_salary,
        'country_Germany': country_ge,
        'country_Spain': country_sp
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error("❌ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")

    st.metric("📉 Churn Probability", f"{probability*100:.2f}%")

    trend = pd.DataFrame({"Day": pd.date_range(end=pd.Timestamp.today(), periods=7),
                          "Probability": np.linspace(probability - 0.1, probability + 0.05, 7).clip(0, 1)})

    st.subheader("📈 Churn Probability Trend (Simulated)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=trend, x="Day", y="Probability", marker="o", color="#00c3ff", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Predicted Churn Risk Trend Over Time")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    st.subheader("📊 Customer Demographics Overview")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df['age'], bins=20, kde=True, color='#00c3ff', ax=ax1)
        ax1.set_title("Age Distribution")
        ax1.set_xlabel("Age Group")
        ax1.set_ylabel("Number of Customers")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        churn_by_gender = df.groupby('gender')['churn'].mean().reset_index()
        sns.barplot(data=churn_by_gender, x='gender', y='churn', palette='coolwarm', ax=ax2)
        ax2.set_title("Churn Rate by Gender")
        ax2.set_xlabel("Gender")
        ax2.set_ylabel("Churn")
        st.pyplot(fig2)