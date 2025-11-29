# 🏥 Predictive Health Analytics & Clinical NLP Platform

## 📌 Project Overview
This project is an end-to-end Healthcare AI platform designed to assist medical professionals in assessing patient risk. It bridges the gap between **Structured Data** (Vitals, Lab Results) and **Unstructured Data** (Clinical Notes).

The system ingests **FHIR-standard** patient data, predicts cardiac risk using Machine Learning, explains the prediction using **SHAP (Explainable AI)**, and extracts medical entities from doctor's notes using **BioBERT (NLP)**.

![Project Screenshot](https://via.placeholder.com/800x400?text=Dashboard+Screenshot+Placeholder)
*(Replace this link with your actual screenshot after uploading)*

## 🚀 Key Features
* **FHIR Data Ingestion:** automated parsing of complex nested JSON patient records (Synthea dataset).
* **Predictive Modeling:** Random Forest Classifier to assess heart disease risk based on vitals (Age, BMI, BP, Cholesterol).
* **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** waterfall plots to visualize *why* a specific patient was flagged as high-risk.
* **Clinical NLP:** Named Entity Recognition (NER) using a fine-tuned BERT model (`d4data/biomedical-ner-all`) to extract Symptoms, Diseases, and Medications from unstructured text.
* **Interactive Dashboard:** Built with Streamlit for real-time interaction.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Explainability:** SHAP
* **NLP / Deep Learning:** Hugging Face Transformers, PyTorch
* **Data Standard:** HL7 FHIR

## 📂 Project Structure
```text
├── app.py                     # Main dashboard application
├── data_ingestion.py          # Script to parse FHIR JSONs into CSV
├── processed_patient_data.csv # Cleaned dataset used for ML training
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation