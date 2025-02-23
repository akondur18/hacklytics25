import streamlit as st
from datetime import datetime

def render_initial_form():
    """
    Renders the initial patient information form.
    Returns True if form is submitted successfully.
    """
    st.title("Initial Patient Information 📋")
    
    with st.form(key="initial_patient_info"):
        # Personal Information Section
        st.header("Personal Information")
        
        # Basic Information
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
        
        # Medical Information
        st.header("Medical Information")
        diagnosis_type = st.selectbox(
            "Initial Diagnosis Type",
            ["Luminal A", "Luminal B", "HER2-enriched", "Triple Negative", "Not sure"]
        )
        
        family_history = st.checkbox("Family History of Breast Cancer")
        
        # Submit button must be the last item in the form
        submit_button = st.form_submit_button("Submit Information")

        if submit_button:
            if not name or not email or not phone:
                st.error("Please fill in all required fields")
                return False
            
            # Store the data in session state
            st.session_state.patient_info = {
                "name": name,
                "email": email,
                "phone": phone,
                "diagnosis_type": diagnosis_type,
                "family_history": family_history,
                "submission_date": datetime.now().isoformat()
            }
            st.session_state.initial_form_submitted = True
            
            st.success("Information saved successfully!")
            return True
            
        return False