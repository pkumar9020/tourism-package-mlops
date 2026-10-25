"""
app.py - Streamlit front end for the Wellness Tourism Package prediction model.

How it fits into the MLOps pipeline
-----------------------------------
GitHub Actions trains the model and COMMITS the best pipeline to
tourism_project/deployment/best_tourism_model_v1.joblib. Streamlit Community
Cloud watches the repository, so every new commit automatically redeploys
this app with the latest model - no manual step needed.

What the app does
-----------------
1. Loads the committed model (preprocessing + XGBoost in one pipeline).
2. Collects customer details from the sales agent through a form.
3. Saves the inputs into a pandas DataFrame with exactly the columns the
   model was trained on.
4. Predicts the probability that the customer buys the package and shows a
   clear recommendation ("contact" / "deprioritise").
5. Optional: scores a whole CSV of customers at once (batch mode).

Run locally:  streamlit run tourism_project/deployment/app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths - resolved relative to THIS file so the app works no matter which
# folder Streamlit is launched from (Streamlit Cloud runs from the repo root).
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = APP_DIR / "model_metrics.json"

# Feature order expected by the trained pipeline (same as prep.py output)
FEATURE_COLUMNS = [
    "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation",
    "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳",
                   layout="wide")


# ---------------------------------------------------------------------------
# Load the model once and cache it across user interactions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained scikit-learn pipeline committed by the CI pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    """Load the evaluation metrics written by train.py (if present)."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


if not MODEL_PATH.exists():
    st.error("Model file not found. Run the GitHub Actions pipeline first so "
             "it commits best_tourism_model_v1.joblib to the deployment folder.")
    st.stop()

model = load_model()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Sidebar: model information and decision threshold
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model information")
    st.write("**Algorithm:** XGBoost (tuned with GridSearchCV)")
    if metrics:
        test = metrics["metrics"]["test"]
        st.metric("Test F1 score", f"{test['f1']:.2f}")
        st.metric("Test recall (buyers found)", f"{test['recall']:.0%}")
        st.metric("Test ROC-AUC", f"{test['roc_auc']:.2f}")
        st.caption(f"Trained from commit: {metrics.get('git_commit', 'n/a')[:7]}")
    st.divider()
    # A lower threshold contacts more customers (higher recall, lower precision)
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Customers with a purchase probability at or "
                               "above this value are flagged for contact.")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🧳 Wellness Tourism Package - Purchase Predictor")
st.write("Enter the customer's details to estimate how likely they are to buy "
         "the new **Wellness Tourism Package** before the sales team "
         "contacts them.")

tab_single, tab_batch = st.tabs(["Single customer", "Batch scoring (CSV)"])

with tab_single:
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        # --- Customer profile ---------------------------------------------
        with col1:
            st.subheader("Customer profile")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox(
                "Marital status", ["Married", "Single", "Divorced", "Unmarried"])
            occupation = st.selectbox(
                "Occupation",
                ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            designation = st.selectbox(
                "Designation",
                ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            monthly_income = st.number_input(
                "Monthly income", min_value=0.0, value=23000.0, step=500.0)

        # --- Travel preferences -------------------------------------------
        with col2:
            st.subheader("Travel preferences")
            city_tier = st.selectbox("City tier", [1, 2, 3],
                                     help="Tier 1 = most developed cities")
            persons = st.number_input("Number of persons visiting",
                                      min_value=1, max_value=10, value=3)
            children = st.number_input("Children (below 5) visiting",
                                       min_value=0, max_value=5, value=1)
            trips = st.number_input("Average trips per year",
                                    min_value=0, max_value=30, value=3)
            property_star = st.selectbox("Preferred hotel rating", [3, 4, 5])
            passport = st.radio("Holds a valid passport?", ["No", "Yes"],
                                horizontal=True)
            own_car = st.radio("Owns a car?", ["No", "Yes"], horizontal=True)

        # --- Sales interaction --------------------------------------------
        with col3:
            st.subheader("Sales interaction")
            contact = st.selectbox("Type of contact",
                                   ["Self Enquiry", "Company Invited"])
            product = st.selectbox(
                "Product pitched",
                ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            duration = st.number_input("Duration of pitch (minutes)",
                                       min_value=0.0, max_value=200.0,
                                       value=15.0)
            followups = st.number_input("Number of follow-ups",
                                        min_value=0, max_value=10, value=3)
            satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

        submitted = st.form_submit_button("Predict purchase likelihood",
                                          type="primary")

    if submitted:
        # Save the inputs into a single-row DataFrame using the training schema
        input_df = pd.DataFrame([{
            "Age": age,
            "TypeofContact": contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons,
            "NumberOfFollowups": followups,
            "ProductPitched": product,
            "PreferredPropertyStar": property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": trips,
            "Passport": 1 if passport == "Yes" else 0,
            "PitchSatisfactionScore": satisfaction,
            "OwnCar": 1 if own_car == "Yes" else 0,
            "NumberOfChildrenVisiting": children,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }])[FEATURE_COLUMNS]

        st.subheader("Input data sent to the model")
        st.dataframe(input_df, hide_index=True, use_container_width=True)

        # Predict probability of purchase (class 1)
        probability = float(model.predict_proba(input_df)[0, 1])

        st.subheader("Prediction")
        st.progress(probability, text=f"Purchase probability: {probability:.1%}")
        if probability >= threshold:
            st.success("✅ **Likely to purchase** - prioritise this customer "
                       "for the Wellness Package campaign.")
        else:
            st.warning("⚠️ **Unlikely to purchase** - deprioritise or nurture "
                       "with lower-cost channels.")

with tab_batch:
    st.write("Upload a CSV containing the 18 feature columns "
             f"(`{', '.join(FEATURE_COLUMNS)}`). Extra columns are ignored.")
    uploaded = st.file_uploader("Customer CSV", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            features = batch_df[FEATURE_COLUMNS].copy()
            features["Gender"] = features["Gender"].replace({"Fe Male": "Female"})
            batch_df["PurchaseProbability"] = model.predict_proba(features)[:, 1]
            batch_df["Contact"] = batch_df["PurchaseProbability"] >= threshold
            batch_df = batch_df.sort_values("PurchaseProbability", ascending=False)
            st.write(f"{int(batch_df['Contact'].sum())} of {len(batch_df)} "
                     "customers flagged for contact.")
            st.dataframe(batch_df, hide_index=True, use_container_width=True)
            st.download_button("Download scored customers",
                               batch_df.to_csv(index=False),
                               "scored_customers.csv", "text/csv")
