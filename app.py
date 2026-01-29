import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# Page config & title
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Accident Severity Predictor", layout="centered")

st.title("Ethiopia Road Accident Severity Predictor")
st.markdown("Enter basic accident details to get a predicted severity level (demo version)")

# ─────────────────────────────────────────────────────────────
# User selects model
# ─────────────────────────────────────────────────────────────
model_options = {
    "Logistic Regression": "model_lr.pkl",
    "Random Forest": "model_rf_best.pkl",
    "Gradient Boosting": "model_gb_best.pkl"
}

selected_model_name = st.selectbox(
    "Choose model",
    options=list(model_options.keys()),
    index=2,  # Default to Gradient Boosting
    help="Select which model to use for prediction"
)

# ─────────────────────────────────────────────────────────────
# Load model function
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.info("Run `python src/train.py` first to generate models.")
        st.stop()
    return joblib.load(path)

# Resolve path
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, model_options[selected_model_name])
model = load_model(model_path)

# ─────────────────────────────────────────────────────────────
# User inputs
# ─────────────────────────────────────────────────────────────
st.subheader("Accident Information")
col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Hour of the day", 0, 23, 14, help="Time when the accident occurred")
    num_vehicles = st.number_input(
        "Number of vehicles involved",
        min_value=1, max_value=10, value=2,
        help="How many vehicles were part of the accident"
    )

with col2:
    num_casualties = st.number_input(
        "Number of casualties",
        min_value=1, max_value=20, value=1,
        help="Number of people injured or killed"
    )

cause = st.selectbox(
    "Main cause of the accident",
    ["No distancing", "Changing lane", "Overspeed", "Wrong overtaking", "Other"],
    help="What was reported as the primary cause"
)

# ─────────────────────────────────────────────────────────────
# Prepare input like training data
# ─────────────────────────────────────────────────────────────
try:
    train_X_path = os.path.join(base_dir, "data", "train_X.csv")
    if not os.path.exists(train_X_path):
        raise FileNotFoundError(f"Cannot find {train_X_path}")
    
    train_columns = pd.read_csv(train_X_path, nrows=0).columns.tolist()
    
    input_row = pd.DataFrame(np.zeros((1, len(train_columns))), columns=train_columns)

    # Fill the actual user inputs
    if 'Hour' in input_row.columns:
        input_row['Hour'] = hour
    if 'Number_of_vehicles_involved' in input_row.columns:
        input_row['Number_of_vehicles_involved'] = num_vehicles
    if 'Number_of_casualties' in input_row.columns:
        input_row['Number_of_casualties'] = num_casualties
    
    # Optional: if you saved label encoders, you can encode cause properly
    # Otherwise leave zero (first category)
    if 'Cause_of_accident' in input_row.columns:
        input_row['Cause_of_accident'] = 0

except Exception as e:
    st.error("Cannot prepare input data correctly")
    st.code(str(e))
    st.stop()

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
if st.button("Predict Severity", type="primary", use_container_width=True):
    with st.spinner("Making prediction..."):
        try:
            pred = model.predict(input_row)[0]
            prob = model.predict_proba(input_row)[0]

            labels = {0: "Fatal", 1: "Serious", 2: "Slight"}
            severity_text = labels.get(pred, "Unknown")

            st.success(f"**Predicted Severity: {severity_text}**", icon="🚨")
            st.markdown("**Probability distribution:**")
            for i, p in enumerate(prob):
                label = labels.get(i, f"Class {i}")
                st.progress(p, text=f"{label}: **{p:.1%}**")

        except ValueError as ve:
            st.error("Feature mismatch error")
            st.info("The input columns/order don't match what the model was trained on.")
            with st.expander("Technical details"):
                st.code(str(ve))
        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

# ─────────────────────────────────────────────────────────────
# Footer / info
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ This is a simplified demonstration only. "
    "Most features are set to default/zero values. "
    "Real-world use would require collecting values for all ~30 features."
)
st.caption("Project based on Addis Ababa Road Traffic Accidents dataset")
