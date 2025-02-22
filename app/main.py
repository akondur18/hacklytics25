# main.py (Frontend + Integration Hub)
import streamlit as st
import boto3
from datetime import datetime
import openai
import pandas as pd
import base64

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

# Medical History Form
def input_form():
    st.title("Patient Intake Form 🩺")
    
    with st.form("medical_history"):
        st.header("Personal Information")
        age = st.number_input("Age", 5, 100)
        gender = st.radio("Gender", ["Female", "Male", "Other"])
        
        st.header("Medical History")
        family_history = st.checkbox("Family history of breast cancer")
        pregnancies = st.number_input("Number of pregnancies", 0, 10)
        hormonal_therapy = st.checkbox("Previous hormonal therapy")
        
        st.header("Blood Test Results")
        blood_file = st.file_uploader("Upload blood test (TXT)", type="txt")
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            # Store in AWS
            # save_to_dynamo({
            #     "email": st.session_state.user_email,
            #     "age": age,
            #     "gender": gender,
            #     "family_history": family_history,
            #     "blood_test": process_bloodtest(blood_file),
            #     "timestamp": datetime.now().isoformat()
            # })
            
            # Get ML Prediction
            cancer_type = get_ml_prediction(age, gender, family_history, blood_file)
            
            # Generate Treatment Plan
            treatment = generate_treatment_plan(cancer_type)
            
            st.session_state.results = {
                "type": cancer_type,
                "treatment": treatment
            }
            st.rerun()

# GenAI Treatment Planner
def generate_treatment_plan(cancer_type):
    openai.api_key = st.secrets["OPENAI_KEY"]
    prompt = f"""Create a detailed breast cancer treatment plan for {cancer_type} type cancer considering:
    - Latest NCCN guidelines
    - Targeted therapies
    - Lifestyle recommendations
    - Clinical trial options"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# AWS Functions
def save_to_dynamo(record):
    table = dynamodb.Table(DYNAMO_TABLE)
    table.put_item(Item=record)

def process_bloodtest(file):
    s3_key = f"blood_tests/{st.session_state.user_email}/{datetime.now().isoformat()}.txt"
    s3.upload_fileobj(file, S3_BUCKET, s3_key)
    return s3_key

# Main App Flow
# if not st.session_state.authenticated:
#     auth_page()
# else:
if 'results' in st.session_state:
    st.title("Your Results 🧪")
    st.subheader(f"Predicted Cancer Type: {st.session_state.results['type']}")
    st.subheader("Personalized Treatment Plan")
    st.markdown(st.session_state.results['treatment'])
        
    if st.button("New Analysis"):
        del st.session_state.results
        st.rerun()
else:
    input_form()

# Aditi's Dashboard (Separate dashboard.py)
# Would query DynamoDB and create visualizations