# app.py
# Thunderstorm Prediction Web App using Streamlit

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Thunderstorm Prediction",
    page_icon="⛈️",
    layout="centered"
)

st.title("⛈️ Thunderstorm Prediction App")
st.write("This web app predicts whether a thunderstorm will occur based on weather data.")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("merged_df_all12k.csv")

try:
    df = load_data()
except:
    st.error("❌ Dataset not found. Keep merged_df_all12k.csv in the same folder.")
    st.stop()

# ================= PREPARE DATA =================
X = df.drop(columns=["TH"])
y = df["TH"]

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# ================= TRAIN MODEL =================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ================= USER INPUT =================
st.subheader("📥 Enter Weather Parameters")

user_input = {}
for col in X.columns:
    user_input[col] = st.slider(
        col,
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )

input_df = pd.DataFrame([user_input])

# ================= PREDICTION =================
if st.button("🔮 Predict Thunderstorm"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚡ Thunderstorm Expected\n\nProbability: {probability*100:.2f}%")
    else:
        st.success(f"☀️ No Thunderstorm Expected\n\nProbability: {probability*100:.2f}%")

st.markdown("---")
st.caption("Developed by Nandini Zurunge")
