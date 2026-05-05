import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load saved files
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")
accuracy = joblib.load("accuracy.pkl")
importance = joblib.load("importance.pkl")
y_test = joblib.load("y_test.pkl")
y_pred = joblib.load("y_pred.pkl")

st.title("🏠 House Price Prediction Dashboard")

st.markdown("### Enter House Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sqft)", 500,10000,2000)
    bedrooms = st.slider("Bedrooms",1,6,3)
    bathrooms = st.slider("Bathrooms",1,4,2)

with col2:
    stories = st.slider("Stories",1,4,2)
    parking = st.slider("Parking Spaces",0,3,1)
    airconditioning = st.selectbox("Air Conditioning",[0,1])

if st.button("Predict Price"):

    features = np.array([[area,bedrooms,bathrooms,stories,parking,airconditioning]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    st.success(f"💰 Predicted Price: ₹ {prediction[0]:,.2f}")

# -----------------------------

st.markdown("### Model Performance")

st.write(f"R² Score: **{accuracy:.2f}**")

# -----------------------------
st.markdown("### Actual vs Predicted Prices")

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, label="Predicted vs Actual")

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--",
         color="red",
         label="Perfect Prediction")

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.legend()
plt.grid(True)


st.pyplot(plt)

# -----------------------------

st.markdown("### Feature Importance")

plt.figure(figsize=(6,4))

importance.sort_values().plot(kind="barh")

plt.xlabel("Importance")
plt.title("Feature Importance")

st.pyplot(plt)