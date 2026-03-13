import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="wide",
    page_icon="🚗"
)

st.title("🚗 Ford Car Price Prediction System")

# Load models
model = pickle.load(open("models/polynomial_model.pkl","rb"))
poly = pickle.load(open("models/poly_transformer.pkl","rb"))
label_model = pickle.load(open("models/label_encoder.pkl","rb"))
trans_cols = pickle.load(open("models/transmission_cols.pkl","rb"))
fuel_cols = pickle.load(open("models/fuel_cols.pkl","rb"))

scores = pd.read_csv("reports/model_scores.csv")
df = pd.read_csv("data/ford.csv")

tab1,tab2,tab3 = st.tabs([
"💰 Price Prediction",
"📊 Data Insights",
"📈 Model Performance"
])

# -------------------
# Prediction Tab
# -------------------

with tab1:

    st.header("Predict Ford Car Price")

    col1,col2 = st.columns(2)

    with col1:

        model_name = st.selectbox("Model",label_model.classes_)

        year = st.slider(
            "Year",
            2000,
            2020,
            2018
        )

        mileage = st.number_input(
            "Mileage",
            0,
            200000,
            10000
        )

    with col2:

        tax = st.number_input("Tax",0,500,150)

        mpg = st.number_input("MPG",10.0,100.0,50.0)

        engine = st.number_input("Engine Size",1.0,5.0,1.5)

    transmission = st.selectbox(
        "Transmission",
        [c.split("_")[1] for c in trans_cols]
    )

    fuel = st.selectbox(
        "Fuel Type",
        [c.split("_")[1] for c in fuel_cols]
    )

    if st.button("Predict Price"):

        model_encoded = label_model.transform([model_name])[0]

        data = {
            "model":[model_encoded],
            "year":[year],
            "mileage":[mileage],
            "tax":[tax],
            "mpg":[mpg],
            "engineSize":[engine]
        }

        df_input = pd.DataFrame(data)

        for col in trans_cols:
            df_input[col] = 1 if col.split("_")[1]==transmission else 0

        for col in fuel_cols:
            df_input[col] = 1 if col.split("_")[1]==fuel else 0

        X_poly = poly.transform(df_input)

        prediction = model.predict(X_poly)

        st.success(
            f"Estimated Price: £{prediction[0]:,.2f}"
        )

# -------------------
# EDA Tab
# -------------------

with tab2:

    st.header("Dataset Insights")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Price Distribution")

    st.bar_chart(df["price"])

    st.subheader("Average Price by Year")

    st.line_chart(
        df.groupby("year")["price"].mean()
    )

    st.subheader("Average Price by Model")

    st.bar_chart(
        df.groupby("model")["price"].mean()
    )

# -------------------
# Model Performance
# -------------------

with tab3:

    st.header("Model Comparison")

    st.dataframe(scores)

    st.subheader("R2 Score Comparison")

    st.bar_chart(
        scores.set_index("Model")["Test_R2"]
    )

    st.subheader("RMSE Comparison")

    st.bar_chart(
        scores.set_index("Model")["RMSE"]
    )