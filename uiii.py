import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        # .stApp {
        #     background: linear-gradient(135deg, #f7f9fc 0%, #eef7ff 100%);
        # }
        .title {
            font-weight: 800;
            font-size: 2.1rem;
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .card {
            background: white;
            border-radius: 1.2rem;
            padding: 1rem 1.2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.06);
            border: 1px solid rgba(2,132,199,0.10);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🏠 House Price Predictor (in Lacs)</div>', unsafe_allow_html=True)

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    with open("Model.pkl", "rb") as f:
        model = pickle.load(f)

    scaler = None
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        pass

    # Feature order from model itself
    try:
        X_cols = list(model.feature_names_in_)
    except AttributeError:
        X_cols = []

    # encoders for categorical cols if needed
    encoders = {}
    if os.path.exists("train.csv"):
        df = pd.read_csv("train.csv")
        for col in ["POSTED_BY", "BHK_OR_RK"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

    return model, scaler, encoders, X_cols

model, scaler, encoders, X_cols = load_artifacts()

# ---------- Input form ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Enter Property Details")

user_inputs = {}

for col in X_cols:
    if col == "BHK_NO.":
        user_inputs[col] = st.number_input("BHK (Number of Bedrooms)", min_value=1, max_value=20, value=2, step=1)
    elif col == "SQUARE_FT":
        user_inputs[col] = st.number_input("Square Feet", min_value=100, max_value=20000, value=1000, step=50)
    elif col in ["UNDER_CONSTRUCTION","RERA","READY_TO_MOVE","RESALE"]:
        user_inputs[col] = st.selectbox(col, options=[0,1], index=1)
    elif col in ["POSTED_BY","BHK_OR_RK"]:
        if col in encoders:
            classes = encoders[col].classes_.tolist()
            choice = st.selectbox(col, options=classes)
            user_inputs[col] = choice
        else:
            user_inputs[col] = st.text_input(col, "Owner")
    else:
        # generic numeric input
        user_inputs[col] = st.number_input(col, value=0.0)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Prepare single-row DataFrame ----------
def build_input_df(user_inputs: dict) -> pd.DataFrame:
    row = {}
    for col in X_cols:
        if col in ["POSTED_BY","BHK_OR_RK"] and col in user_inputs:
            if col in encoders:
                row[col] = encoders[col].transform([str(user_inputs[col])])[0]
            else:
                row[col] = 0
        else:
            row[col] = user_inputs.get(col, 0)
    return pd.DataFrame([row], columns=X_cols)

# ---------- Predict ----------
if st.button("🔮 Predict Price (Lacs)"):
    X_input = build_input_df(user_inputs)

    if scaler is not None:
        try:
            X_trans = scaler.transform(X_input)
        except Exception as e:
            st.warning("Scaler shape mismatch; using raw inputs for prediction.")
            X_trans = X_input.values
    else:
        X_trans = X_input.values

    try:
        y_pred = model.predict(X_trans)
        st.success(f"Estimated Price: {float(y_pred[0]):,.2f} Lacs")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
