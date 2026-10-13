
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Telemarketing Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("telemarketing_model.pkl")

model = load_model()

# Application title
st.title("🏦 Telemarketing Campaign Prediction")

st.write(
    "This application predicts whether a bank customer "
    "is likely to subscribe to a fixed deposit."
)

st.divider()

st.subheader("Enter Customer Details")

# Customer inputs
age = st.number_input(
    "Customer Age",
    min_value=17,
    max_value=95,
    value=35
)

job = st.selectbox(
    "Occupation",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    ["married", "single", "divorced"]
)

education = st.selectbox(
    "Education Level",
    [
        "Primary_Education",
        "Secondary_Education",
        "Tertiary_Education",
        "Professional_Education"
    ]
)

default_credit = st.selectbox(
    "Credit Default",
    ["no", "yes"]
)

housing_loan = st.selectbox(
    "Housing Loan",
    ["no", "yes"]
)

personal_loan = st.selectbox(
    "Personal Loan",
    ["no", "yes"]
)

# Prediction button
if st.button("Predict Subscription", type="primary"):

    customer_data = pd.DataFrame([{
        "Age": age,
        "Job": job,
        "Marital_Status": marital_status,
        "Education": education,
        "Default_Credit": default_credit,
        "Housing_Loan": housing_loan,
        "Personal_Loan": personal_loan
    }])

    prediction = model.predict(customer_data)[0]

    probability = model.predict_proba(
        customer_data
    )[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "Prediction: Customer is likely to subscribe."
        )
    else:
        st.warning(
            "Prediction: Customer is unlikely to subscribe."
        )

    st.metric(
        "Estimated Subscription Probability",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "This prediction is based on a machine learning "
        "model and should not be treated as a certainty."
    )

st.divider()

st.caption(
    "Telemarketing Campaign Prediction | "
    "Machine Learning Capstone Project"
)
