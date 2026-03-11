import streamlit as st
import numpy as np
import pickle

with open("price_model.pkl", "rb") as f:
    model = pickle.load(f)


st.set_page_config(page_title="Price Prediction App")

st.title("🛒 Retail Price Prediction")
st.write("Enter transaction details to predict price")

quantity = st.number_input("Quantity", min_value=1)
customer_id = st.number_input("Customer ID", min_value=1)
invoice = st.number_input("Invoice Number", min_value=1)
country = st.number_input("Country Code", min_value=1)
year = st.number_input("Year", min_value=2000, max_value=2030)
month = st.number_input("Month", min_value=1, max_value=12)
day = st.number_input("Day", min_value=1, max_value=31)

if st.button("Predict Price"):

    features = np.array([[quantity, customer_id, invoice, country, year, month, day]], dtype=float)

    prediction = model.predict(features)

    st.success(f"Predicted Price: {prediction[0]:.2f}")
