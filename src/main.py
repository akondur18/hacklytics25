import streamlit as st
from datetime import datetime
import boto3
import pandas as pd
import openai
import json
from components.auth import render_auth_page
from components.forms.initial_form import render_initial_form
from components.forms.monthly_form import render_monthly_form
from database.dynamo_ops import save_patient_info, save_monthly_tracking
from utils.data_processing import validate_patient_data, validate_monthly_data
from components.dashboard import display_patient_history

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'initial_form_submitted' not in st.session_state:
    st.session_state.initial_form_submitted = False
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None

def predict_risk(age, gender, er_status, pr_status, her2_status, tumor_stage, hist_grade):
    """Predict cancer subtype based on markers."""
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

def generate_treatment_plan(cancer_type, blood_file=None):
    """Generate AI-powered treatment recommendations."""
    try:
        # Check if OpenAI API key is configured
        openai_key = st.secrets.get("OPENAI_KEY")
        if not openai_key:
            st.error("OpenAI API key not found in secrets. Please configure your .streamlit/secrets.toml file.")
            return None
            
        openai.api_key = openai_key
        
        patient_data = {
            "cancer_type": cancer_type,
            "blood_file": blood_file.name if blood_file else "Not provided"
        }
        
        prompt = f"""
        Based on the following patient data and model prediction, analyze the breast cancer type:
        {json.dumps(patient_data, indent=2)}
        
        Provide a detailed analysis of:
        1. Key risk factors for Infiltrating Ductal Carcinoma, Infiltrating Lobular Carcinoma, or Mucinous Carcinoma
        2. Confidence in the breast cancer type diagnosis
        3. Additional tests or information needed
        
        Recommendations for:
        1. Potential medical treatments and clinical trials
        2. Lifestyle modifications and monitoring
        3. Specialists to consult
        
        Note: Frame this as suggestions to discuss with healthcare providers.
        """
        
        client = openai.OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a medical AI assistant specializing in breast cancer analysis and treatment recommendations."
                },
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error("Error generating treatment plan. Please try again later.")
        print(f"Treatment plan generation error: {str(e)}")  # Replace with proper logging
        return None

def save_analysis_results(patient_id, cancer_type, treatment_plan):
    """Save analysis results to the database."""
    try:
        # Initialize DynamoDB client
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('PatientAnalysis')
        
        # Prepare analysis data
        analysis_data = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'cancer_type': cancer_type,
            'treatment_plan': treatment_plan
        }
        
        # Save to DynamoDB
        table.put_item(Item=analysis_data)
        
    except Exception as e:
        raise Exception(f"Error saving analysis results: {str(e)}")

def render_analysis_section():
    """Render the analysis and treatment recommendation section."""
    st.header("Analysis & Recommendations")
    
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            er_status = st.selectbox("ER Status", ["Positive", "Negative"])
            pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
        
        with col2:
            her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
            tumor_stage = st.selectbox("Tumor Stage", ["I", "II", "III", "IV"])
            hist_grade = st.selectbox("Histological Grade", ["1", "2", "3"])
            blood_file = st.file_uploader("Upload Blood Work Results (optional)", type=["pdf", "csv", "xlsx"])
        
        submitted = st.form_submit_button("Generate Analysis")
        
        if submitted:
            with st.spinner("Analyzing data and generating recommendations..."):
                # Predict cancer subtype
                cancer_type = predict_risk(
                    age, gender, er_status, pr_status, 
                    her2_status, tumor_stage, hist_grade
                )
                
                st.subheader("Predicted Cancer Subtype")
                st.write(cancer_type)
                
                # Generate treatment plan
                treatment_plan = generate_treatment_plan(cancer_type, blood_file)
                if treatment_plan:
                    st.subheader("AI-Generated Treatment Recommendations")
                    st.write(treatment_plan)
                    
                    # Save analysis to database
                    if st.session_state.patient_id:
                        try:
                            save_analysis_results(
                                st.session_state.patient_id,
                                cancer_type,
                                treatment_plan
                            )
                            st.success("Analysis results saved successfully")
                        except Exception as e:
                            st.error("Error saving analysis results")
                            print(f"Database error: {str(e)}")  # Replace with proper logging

def main():
    st.set_page_config(
        page_title="BRCA Care Portal",
        page_icon="🩺",
        layout="wide"
    )

    st.markdown("""
    <style>
        .stApp {
            background-color: #FFDBE0;
        }
        
        /* For tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #FFDBE0;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #FFDBE0;
        }
        
        /* For sidebar */
        .css-1d391kg {
            background-color: #FFDBE0;
        }
        
        /* For content area */
        .block-container {
            background-color: #FFDBE0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Check authentication
    if not st.session_state.authenticated:
        render_auth_page()
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Initial Form", "Monthly Tracking", "Analysis", "View History"],
        disabled=not st.session_state.authenticated
    )

    if page == "Initial Form":
        if not st.session_state.initial_form_submitted:
            render_initial_form()
        else:
            st.info("Initial form already submitted. You can proceed to monthly tracking.")
            if st.button("View Monthly Tracking"):
                st.session_state.page = "Monthly Tracking"
                st.rerun()

    elif page == "Monthly Tracking":
        if not st.session_state.initial_form_submitted:
            st.warning("Please complete the initial form first.")
            if st.button("Go to Initial Form"):
                st.session_state.page = "Initial Form"
                st.rerun()
        else:
            render_monthly_form()

    elif page == "Analysis":
        if not st.session_state.initial_form_submitted:
            st.warning("Please complete the initial form first.")
            if st.button("Go to Initial Form"):
                st.session_state.page = "Initial Form"
                st.rerun()
        else:
            render_analysis_section()

    elif page == "View History":
        if not st.session_state.initial_form_submitted:
            st.warning("Please complete the initial form first.")
        else:
            display_patient_history()

if __name__ == "__main__":
    main()