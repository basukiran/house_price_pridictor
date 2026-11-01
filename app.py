import os
import pickle as pkl
import base64

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

@st.cache_resource
def load_model(model_path, encoded_path, modified_time):
    if os.path.exists(encoded_path):
        with open(encoded_path, "rt", encoding="ascii") as f:
            payload = base64.b64decode(f.read())
        return pkl.loads(payload)
    with open(model_path, "rb") as f:
        return pkl.load(f)


model_path = "housing_model.pkl"
encoded_model_path = "housing_model.pkl.b64"
model_mtime = os.path.getmtime(encoded_model_path if os.path.exists(encoded_model_path) else model_path)
data = load_model(model_path, encoded_model_path, model_mtime)
pipe = data["pipeline"]

options = {
    "MS Zoning": ["RL", "RM", "FV", "RH", "C (all)"],
    "Bldg Type": ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"],
    "House Style": ["1Story", "1.5Fin", "2Story", "1.5Unf", "SFoyer", "SLvl"],
    "Exter Qual": ["Ex", "Gd", "TA", "Fa", "Po"],
    "Bsmt Qual": ["Ex", "Gd", "TA", "Fa", "Po"],
    "Heating QC": ["Ex", "Gd", "TA", "Fa", "Po"],
    "Garage Qual": ["Ex", "Gd", "TA", "Fa", "Po"],
    "Sale Type": ["WD", "New", "COD", "Con", "Oth"],
}
numeric = ["Overall Qual", "Overall Cond", "Year Built", "Bedroom AbvGr", "TotRms AbvGrd", "Gr Liv Area"]

# The full pipeline uses all original input fields, rather than only the
# reduced set retained for SHAP interpretation.
important_fields = list(options) + numeric
core_fields = ["Gr Liv Area", "Overall Qual", "Year Built", "Bsmt Qual", "Bedroom AbvGr"]

profiles = {
    "Custom": {},
    "Starter home": {"MS Zoning": "RL", "Bldg Type": "1Fam", "House Style": "1Story", "Overall Qual": 5, "Overall Cond": 5, "Year Built": 1995, "Exter Qual": "TA", "Bsmt Qual": "TA", "Heating QC": "TA", "Bedroom AbvGr": 3, "TotRms AbvGrd": 6, "Garage Qual": "TA", "Sale Type": "WD", "Gr Liv Area": 1200},
    "Family home": {"MS Zoning": "RL", "Bldg Type": "1Fam", "House Style": "2Story", "Overall Qual": 7, "Overall Cond": 5, "Year Built": 2005, "Exter Qual": "Gd", "Bsmt Qual": "Gd", "Heating QC": "Gd", "Bedroom AbvGr": 4, "TotRms AbvGrd": 8, "Garage Qual": "TA", "Sale Type": "WD", "Gr Liv Area": 2100},
    "Newer townhome": {"MS Zoning": "FV", "Bldg Type": "TwnhsE", "House Style": "2Story", "Overall Qual": 7, "Overall Cond": 5, "Year Built": 2010, "Exter Qual": "Gd", "Bsmt Qual": "Gd", "Heating QC": "Ex", "Bedroom AbvGr": 3, "TotRms AbvGrd": 7, "Garage Qual": "Gd", "Sale Type": "New", "Gr Liv Area": 1750},
}

# Preset changes are copied into widget state before the form is drawn.
def apply_profile():
    preset = profiles[st.session_state.profile]
    defaults = {"MS Zoning": "RL", "Bldg Type": "1Fam", "House Style": "1Story", "Overall Qual": 5, "Overall Cond": 5, "Year Built": 2000, "Exter Qual": "TA", "Bsmt Qual": "TA", "Heating QC": "TA", "Bedroom AbvGr": 3, "TotRms AbvGrd": 6, "Garage Qual": "TA", "Sale Type": "WD", "Gr Liv Area": 1500}
    defaults.update(preset)
    for field in important_fields:
        st.session_state["input_" + field] = defaults[field]

st.title("🏠 House Price Predictor")
st.write("Explore how home features affect an estimated sale price.")

profile_col, info_col = st.columns([2, 3])
with profile_col:
    st.selectbox("Start with a sample home", list(profiles), key="profile", on_change=apply_profile)
with info_col:
    st.caption("Sample profiles are starting points. You can adjust any selected feature before predicting.")

with st.form("house_form"):
    st.subheader("Quick estimate")
    st.caption("Start with these five details. The other model inputs use typical defaults unless you add them below.")

    def render_field(field):
        if field in options:
            st.selectbox(field, options[field], key="input_" + field)
        elif field in ("Overall Qual", "Overall Cond"):
            st.slider(field, 1, 10, key="input_" + field, help="1 = low, 10 = excellent")
        elif field == "Year Built":
            st.number_input(field, 1800, 2026, key="input_" + field, step=50)
        elif field == "Bedroom AbvGr":
            st.number_input("Bedrooms", 0, 20, key="input_" + field)
        elif field == "TotRms AbvGrd":
            st.number_input("Total rooms", 1, 30, key="input_" + field)
        elif field == "Gr Liv Area":
            st.number_input("Living area (sq ft)", 200, 10000, key="input_" + field, step=50)

    core_cols = st.columns(2)
    for i, field in enumerate(core_fields):
        with core_cols[i % 2]:
            render_field(field)

    optional_fields = [field for field in important_fields if field not in core_fields]
    with st.expander("More details (optional)"):
        st.caption("Add these if you know them; otherwise the app will use typical values.")
        optional_cols = st.columns(2)
        for i, field in enumerate(optional_fields):
            with optional_cols[i % 2]:
                render_field(field)

    submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

if submitted:
    defaults = {"MS Zoning": "RL", "Bldg Type": "1Fam", "House Style": "1Story", "Overall Qual": 5, "Overall Cond": 5, "Year Built": 2000, "Exter Qual": "TA", "Bsmt Qual": "TA", "Heating QC": "TA", "Bedroom AbvGr": 3, "TotRms AbvGrd": 6, "Garage Qual": "TA", "Sale Type": "WD", "Gr Liv Area": 1500}
    for field in important_fields:
        widget_key = "input_" + field
        if widget_key in st.session_state:
            defaults[field] = st.session_state[widget_key]
    raw = pd.DataFrame([defaults])
    cols = data.get("outlier_columns", numeric)
    raw[cols] = raw[cols].clip(lower=data["lower"], upper=data["upper"], axis=1)
    prediction = float(pipe.predict(raw)[0])
    rmse = float(data["rmse"])
    low, high = max(0, prediction - rmse), prediction + rmse
    st.session_state.prediction_result = {"prediction": prediction, "low": low, "high": high, "rmse": rmse}

result = st.session_state.get("prediction_result")
if result:
    st.divider()
    st.subheader("Your estimate")
    a, b = st.columns([2, 3])
    with a:
        st.metric("Approximate price range", f"${result['low']:,.0f} to ${result['high']:,.0f}", help="Prediction ± test-set RMSE")
        st.caption(f"Model estimate: ${result['prediction']:,.0f}")
    with b:
        st.markdown("**Typical model error range**")
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.9rem'><span>${result['low']:,.0f}</span><span>Estimate ${result['prediction']:,.0f}</span><span>${result['high']:,.0f}</span></div><div style='position:relative;height:14px;background:linear-gradient(90deg,#dbeafe,#2563eb,#dbeafe);border-radius:10px;margin:8px 0 4px'><span style='position:absolute;left:50%;top:-4px;height:22px;border-left:3px solid #102a43'></span></div>", unsafe_allow_html=True)
        st.caption(f"The range uses ±${result['rmse']:,.0f} RMSE, the model's average test error scale. It is approximate, not a guarantee.")

with st.expander("How this estimate works"):
    st.write("The app uses the full linear regression pipeline trained on Ames housing sales. It uses all the home details above. The displayed range is centered on the model prediction and extends by the full model's test RMSE.")
    st.caption("Model inputs: " + ", ".join(important_fields))
