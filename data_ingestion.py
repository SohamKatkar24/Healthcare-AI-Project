import pandas as pd
import json
import os
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = r"D:\Documents\Personal Projects\Healthcare-AI_Project\synthea_sample_data_fhir_r4_sep2019\fhir"
OUTPUT_FILE = "D:\Documents\Personal Projects\Healthcare-AI_Project\processed_patient_data.csv"

# --- LOINC CODES ---
CODES = {
    'bmi': '39156-5',
    'cholesterol': '2093-3',
    'smoking_status': '72166-2'
}

def calculate_age(birthdate_str):
    if not birthdate_str: return 0
    try:
        birth = datetime.strptime(birthdate_str, "%Y-%m-%d")
        today = datetime.today()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except:
        return 0

def process_fhir_files(directory, limit=500):
    patient_records = []
    
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return

    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files. Processing first {limit}...")

    for i, filename in enumerate(files):
        if limit is not None and i >= limit: break
        
        try:
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('resourceType') != 'Bundle': continue

            record = {
                'file': filename,
                'age': None,
                'bmi': None,
                'systolic_bp': None,
                'cholesterol': None,
                'diabetes_history': 0,
                'smoker': 0
            }

            for entry in data.get('entry', []):
                resource = entry.get('resource', {})
                r_type = resource.get('resourceType')
                
                # 1. AGE
                if r_type == 'Patient':
                    record['age'] = calculate_age(resource.get('birthDate'))

                # 2. VITALS (The Robust Way)
                elif r_type == 'Observation':
                    # Get code and display text for safe checking
                    coding = resource.get('code', {}).get('coding', [{}])[0]
                    code = coding.get('code')
                    display = coding.get('display', '').lower()
                    
                    # A. BMI & Cholesterol (Standard Codes)
                    if code == CODES['bmi']:
                        record['bmi'] = resource.get('valueQuantity', {}).get('value')
                    elif code == CODES['cholesterol']:
                        record['cholesterol'] = resource.get('valueQuantity', {}).get('value')

                    # B. Blood Pressure (The "Catch-All" Logic)
                    # Strategy: Look for "Systolic" in the component list OR the main observation
                    
                    # Case 1: BP is split into components (Common in Synthea)
                    if 'component' in resource:
                        for comp in resource['component']:
                            c_display = comp.get('code', {}).get('coding', [{}])[0].get('display', '').lower()
                            if 'systolic' in c_display:
                                record['systolic_bp'] = comp.get('valueQuantity', {}).get('value')
                    
                    # Case 2: BP is a standalone observation (Fallback)
                    elif 'systolic' in display:
                         record['systolic_bp'] = resource.get('valueQuantity', {}).get('value')

                    # C. Smoker Status
                    elif code == CODES['smoking_status']:
                        text = resource.get('valueCodeableConcept', {}).get('text', '').lower()
                        if 'smoker' in text and 'never' not in text:
                            record['smoker'] = 1

                # 3. CONDITIONS
                elif r_type == 'Condition':
                    c_text = resource.get('code', {}).get('text', '').lower()
                    if 'diabetes' in c_text:
                        record['diabetes_history'] = 1

            # Save valid records (must have Age to be useful)
            if record['age'] is not None:
                patient_records.append(record)

        except Exception as e:
            print(f"Skipping {filename}: {e}")

    # --- DATAFRAME CLEANUP ---
    df = pd.DataFrame(patient_records)
    print(f"Extracted {len(df)} raw records.")
    
    if not df.empty:
        # 1. Fix Types (Solves the FutureWarning)
        cols_to_numeric = ['age', 'bmi', 'systolic_bp', 'cholesterol']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. Fill Missing Values (Imputation)
        # We use median because it's less sensitive to outliers than mean
        df['systolic_bp'] = df['systolic_bp'].fillna(120) 
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        df['cholesterol'] = df['cholesterol'].fillna(df['cholesterol'].median())

        # 3. Generate Target Variable
        df['heart_disease_risk'] = df.apply(
            lambda x: 1 if (x['systolic_bp'] > 140) or (x['diabetes_history'] == 1) else 0, axis=1
        )
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"SUCCESS: Saved {len(df)} clean records to {OUTPUT_FILE}")
        
        # Print a sample to verify BP is captured
        print("\n--- Sample Data ---")
        print(df[['age', 'systolic_bp', 'heart_disease_risk']].head())

if __name__ == "__main__":
    process_fhir_files(DATA_DIR, limit=None)