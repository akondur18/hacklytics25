# main.py (Frontend + Integration Hub)
import streamlit as st
import boto3
from datetime import datetime
import openai
import pandas as pd
import base64
import re
import json

# AWS Configuration
# AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
# AWS_SECRET_KEY = st.secrets["AWS_SECRET_SECRET"]
# DYNAMO_TABLE = "brca_users"
# S3_BUCKET = "brca-blood-tests"

# dynamodb = boto3.resource('dynamodb',
#                          aws_access_key_id=AWS_ACCESS_KEY,
#                          aws_secret_access_key=AWS_SECRET_KEY,
#                          region_name='us-east-1')

# s3 = boto3.client('s3',
#                  aws_access_key_id=AWS_ACCESS_KEY,
#                  aws_secret_access_key=AWS_SECRET_SECRET)

# # Initialize session state
# if 'authenticated' not in st.session_state:
#     st.session_state.authenticated = False

# Login/Registration System
def auth_page():
    st.title("BRCA Care Portal 🔒")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("Login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if verify_user(email, password):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        with st.form("Register"):
            new_email = st.text_input("Email")
            new_pass = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create Account")
            if submitted:
                create_user(new_email, new_pass)
                st.success("Account created! Please login")

# medical form 
def input_form():
    st.title("Breast Cancer Risk Assessment Form 🩺")
    
    with st.form("medical_history"):
        st.header("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        gender = st.radio("Gender", ["Female", "Male", "Other"])
        
        st.header("Clinical Biomarkers (from Blood Test)")
        col1, col2, col3 = st.columns(3)
        with col1:
            er_status = st.radio("ER Status", ["Positive", "Negative"])
        with col2:
            pr_status = st.radio("PR Status", ["Positive", "Negative"])
        with col3:
            her2_status = st.radio("HER2 Status", ["Positive", "Negative"])
            
        st.header("Tumor Characteristics")
        tumor_stage = st.selectbox("Tumor Stage", ["I", "II", "III"])
        histology = st.selectbox("Histology Type", [
            "Infiltrating Ductal Carcinoma", 
            "Infiltrating Lobular Carcinoma",
            "Mucinous Carcinoma"
        ])
        
        st.header("Additional Risk Factors")
        family_history = st.checkbox("Family history of breast cancer")
        menarche_age = st.number_input("Age at first menstruation", 8, 20, 12)
        menopause_status = st.checkbox("Post-menopausal")
        
        st.header("Blood Test Upload")
        blood_file = st.file_uploader("Upload formatted blood test (TXT)", type="txt", 
                    help="Upload file with format:\nER: Positive\nPR: Negative\nHER2: Positive")
        
        submitted = st.form_submit_button("Analyze Risk")
        
        if submitted:
            # Process blood test or use manual inputs
            if blood_file:
                biomarkers = parse_bloodtest(blood_file)
            else:
                biomarkers = {
                    'ER': er_status,
                    'PR': pr_status,
                    'HER2': her2_status
                }
            
            # Get prediction
            prediction = predict_risk(
                age=age,
                gender=gender,
                er_status=biomarkers['ER'],
                pr_status=biomarkers['PR'],
                her2_status=biomarkers['HER2'],
                tumor_stage=tumor_stage,
                histology=histology,
                family_history=family_history
            )
                        
            st.session_state.results = {
                "type": prediction,  
                "biomarkers": biomarkers,
                "stage": tumor_stage
            }
            st.rerun()

import re

def parse_bloodtest(file):
    """Robust parser for clinical blood tests with mixed formats"""
    biomarkers = {
        'ER': 0.0, 'PR': 0.0, 'HER2': 0.0,
        'CA153': 0.0, 'EGFR': 0.0, 'KI67': 0.0,
        'BRCA1': 0.0, 'BRCA2': 0.0, 'TP53': 0.0
    }
    
    try:
        content = file.read().decode().split('\n')
        for line in content:
            if ':' in line:
                key_part, value_part = line.split(':', 1)
                key = key_part.strip().upper()
                value = value_part.strip()
                
                # Extract first numerical value using regex
                match = re.search(r"[-+]?\d*\.?\d+", value)
                if match:
                    numerical_value = float(match.group())
                    
                    # Handle percentage conversion
                    if '%' in value:
                        numerical_value /= 100
                else:
                    # Handle textual statuses
                    status_map = {'positive': 1.0, 'negative': 0.0}
                    numerical_value = status_map.get(value.lower(), 0.0)
                
                # Map key variations
                key = key.replace(' ', '').replace('-', '')
                key_mappings = {
                    'CA153': 'CA153', 'KI67': 'KI67',
                    'MUC1': 'MUC1', 'CCND1': 'CCND1'
                }
                key = key_mappings.get(key, key)
                
                if key in biomarkers:
                    biomarkers[key] = numerical_value
                    
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
    
    return biomarkers

def predict_risk(age, gender, er_status, pr_status, her2_status, tumor_stage, histology, family_history):
    """Mock ML prediction function - replace with actual model"""
    # Convert inputs to match Kaggle dataset format
    # Here you would typically:
    # 1. Encode categorical variables
    # 2. Load trained model
    # 3. Make prediction
    
    # Example logic based on biomarkers
    if er_status == "Positive" and pr_status == "Positive" and her2_status == "Negative":
        return "Luminal A"
    elif er_status == "Positive" and pr_status == "Positive" and her2_status == "Positive":
        return "Luminal B"
    elif er_status == "Negative" and pr_status == "Negative" and her2_status == "Positive":
        return "HER2-enriched"
    elif er_status == "Negative" and pr_status == "Negative" and her2_status == "Negative":
        return "Triple Negative"
    else:
        return "Other Subtype"
    
# GenAI Treatment Planner
def generate_treatment_plan(cancer_type, blood_file):
    openai.api_key = st.secrets.get("OPENAI_KEY")
    patient_data = {"cancer type": cancer_type, "blood file": blood_file}
    prompt = f"""
        Based on the following patient data and model prediction, decipher the type of breast cancer that the patient has between
        Infiltrating Ductal Carcinoma, Infiltrating Lobular Carcinoma, Mucinous Carcinoma.
        
        Patient Data:
        {json.dumps(patient_data, indent=2)}
        
        Provide a detailed analysis of:
        1. Key risk factors identified wtih having Infiltrating Ductal Carcinoma, Infiltrating Lobular Carcinoma, or Mucinous Carcinoma
        2. Confidence in the breast cancer type diagnosis
        3. Additional tests or information that might be needed based on diagnosis and blood input
        
        Provide recommendations for:
        1. Potential medical treatments / clinical trial options
        2. Lifestyle modifications, regular monitoring, and follow-up
        4. Additional specialists to consult
        
        Note: Frame this as suggestions to discuss with a healthcare provider.
        """
        
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical AI assistant specializing in diagnosis between three types of cancer: Infiltrating Ductal Carcinoma, Infiltrating Lobular Carcinoma, Mucinous Carcinoma and providing treatment. Provide detailed but accessible analysis."},
            {"role": "user", "content": prompt}
        ]
    )
        
    return response.choices[0].message["content"]

# AWS Functions
def save_to_dynamo(record):
    table = dynamodb.Table(DYNAMO_TABLE)
    table.put_item(Item=record)

def process_bloodtest(file):
    s3_key = f"blood_tests/{st.session_state.user_email}/{datetime.now().isoformat()}.txt"
    s3.upload_fileobj(file, S3_BUCKET, s3_key)
    return s3_key

def safe_float(biomarkers, key):
    """Type-safe float conversion with error handling"""
    try:
        return float(biomarkers.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
        
# Main App 

if 'results' in st.session_state:
    st.title("Clinical Analysis Report")
    
    # Safely get values with fallbacks
    er_value = safe_float(st.session_state.results['biomarkers'], 'ER')
    her2_value = safe_float(st.session_state.results['biomarkers'], 'HER2')
    ki67_value = safe_float(st.session_state.results['biomarkers'], 'KI67')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Biomarkers")
        st.metric("ER Status", 
                 f"{'Positive' if er_value > 1.0 else 'Negative'}",
                 f"{er_value:.2f} fmol/mL")
        st.metric("HER2 Status", 
                 f"{'Positive' if her2_value > 15.0 else 'Negative'}",
                 f"{her2_value:.2f} ng/mL")
    
    with col2:
        st.subheader("Proliferation Markers")
        st.metric("Ki-67 Index", 
                 f"{ki67_value * 100:.1f}%",
                 "High" if ki67_value > 0.2 else "Low")
        st.metric("BRCA1 Mutation Risk",
                 "High" if st.session_state.results['biomarkers']['BRCA1'] < 5.0 else "Low",
                 f"{st.session_state.results['biomarkers']['BRCA1']:.1f} ng/mL")

else:
    input_form()

# Aditi's Dashboard (Separate dashboard.py)
# Would query DynamoDB and create visualizations

# main.py (add this to the end of the file)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Input Form", "Dashboard"])

if page == "Input Form":
    input_form()
elif page == "Dashboard":
    # Import and run the dashboard
    from dashboard import dashboard
    dashboard()