import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline

# --- 1. SETUP & CACHING ---
@st.cache_resource
def load_data_and_model():
    try:
        df = pd.read_csv('processed_patient_data.csv')
    except FileNotFoundError:
        return None, None
        
    # Features (X) and Target (y)
    X = df[['age', 'bmi', 'systolic_bp', 'cholesterol', 'diabetes_history', 'smoker']]
    y = df['heart_disease_risk']

    # Train Model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    return model, X

@st.cache_resource
def load_nlp_model():
    # Load a specific medical NER model from Hugging Face
    # This might take 30-60 seconds to download on the first run
    nlp_ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
    return nlp_ner

# Load resources
model, X_train = load_data_and_model()
nlp_model = load_nlp_model()

# --- 2. SIDEBAR (INPUTS) ---
st.sidebar.title("🏥 MedAI Platform")
st.sidebar.info("Navigation")
app_mode = st.sidebar.radio("Go to:", ["Risk Assessment (Structured)", "Clinical Notes (Unstructured)"])

# --- 3. PAGE 1: RISK ASSESSMENT ---
if app_mode == "Risk Assessment (Structured)":
    st.title("❤️ Cardiac Risk Prediction")
    st.markdown("Predictive analysis using **Random Forest** & **SHAP**.")
    
    if model is None:
        st.error("Data file not found. Please run 'data_ingestion.py' first!")
        st.stop()

    st.divider()
    
    # Input Form
    st.subheader("Patient Vitals")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
    with col2:
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    with col3:
        diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
        smoker = st.selectbox("Smoker?", ["No", "Yes"])

    # Prepare input for model
    input_data = pd.DataFrame({
        'age': [age], 'bmi': [bmi], 'systolic_bp': [systolic_bp], 
        'cholesterol': [cholesterol], 
        'diabetes_history': [1 if diabetes == 'Yes' else 0],
        'smoker': [1 if smoker == 'Yes' else 0]
    })

    if st.button("Analyze Risk"):
        # Prediction
        prob = model.predict_proba(input_data)[0][1]
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Risk Probability", f"{prob:.1%}")
        with c2:
            if prob > 0.5:
                st.error("⚠️ HIGH RISK DETECTED")
            else:
                st.success("✅ LOW RISK")

        # SHAP Waterfall
        st.subheader("🔍 Explainability Analysis")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)
        
        # Handle SHAP shape
        if len(shap_values.shape) == 3:
             explanation = shap_values[0, :, 1]
        else:
             explanation = shap_values[0]
             
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig, bbox_inches='tight')

# --- 4. PAGE 2: CLINICAL NOTES (NLP) ---
elif app_mode == "Clinical Notes (Unstructured)":
    st.title("📝 Clinical Note Analyzer")
    st.markdown("Uses **BioBERT (NER)** to extract symptoms and diseases from text.")
    
    default_text = "Patient is a 50 year old male complaining of severe chest pain and shortness of breath. History of hypertension and diabetes. Prescribed Aspirin."
    text_input = st.text_area("Doctor's Notes:", default_text, height=150)
    
    if st.button("Extract Entities"):
        with st.spinner("Analyzing text..."):
            results = nlp_model(text_input)
            
            st.subheader("Extracted Medical Entities")
            
            # Create a clean table from the raw NLP output
            clean_data = []
            for item in results:
                clean_data.append({
                    "Entity": item['word'],
                    "Type": item['entity_group'],
                    "Confidence": f"{item['score']:.2%}"
                })
            
            if clean_data:
                st.table(pd.DataFrame(clean_data))
            else:
                st.warning("No specific medical entities found.")