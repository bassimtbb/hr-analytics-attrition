import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HR Attrition Predictor", layout="centered")

st.title("HR Analytics – Attrition Prediction")
st.write("Cette application prédit le risque de départ des employés à partir de données RH.")

# Load model
model = joblib.load("models/Random_Forest.joblib")

# Upload CSV
uploaded_file = st.file_uploader("Upload HR dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    # Prepare X
    X = df.drop(columns=["Attrition"], errors="ignore")

    if st.button("Prédire l'attrition"):
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        df["Attrition_Prediction"] = predictions
        df["Attrition_Probability"] = probabilities

        st.subheader("Résultats")
        st.dataframe(df[["Attrition_Prediction", "Attrition_Probability"]].head())

        st.subheader("Analyse du risque")
        high_risk = df["Attrition_Probability"] > 0.6
        st.write("Employés à risque élevé :", int(high_risk.sum()))

        mean_prob = probabilities.mean()
        st.write("Probabilité moyenne d’attrition :", round(mean_prob, 2))

        if mean_prob > 0.5:
            st.warning("⚠️ Risque d’attrition global élevé")
        else:
            st.success("✅ Risque d’attrition global modéré")
