import streamlit as st
from datetime import datetime
import pandas as pd

def process_blood_test_file(uploaded_file):
    """
    Process the uploaded blood test txt file and return the data
    """
    try:
        content = uploaded_file.getvalue().decode()
        return {"blood_test_content": content}
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None

def render_monthly_form():
    """
    Renders the monthly health tracking form.
    Returns the form data if submitted successfully.
    """
    st.title("Monthly Health Tracking 🏥")
    
    with st.form(key="monthly_health_tracking"):
        # Visit Information
        st.header("Visit Information")
        
        visit_date = st.date_input("Visit Date")
        provider_name = st.text_input("Healthcare Provider Name")
        facility_name = st.text_input("Facility Name")
        
        # Health Status
        st.header("Current Health Status")
        
        pain_level = st.slider("Pain Level (0-10)", 0, 10, 0)
        fatigue_level = st.slider("Fatigue Level (0-10)", 0, 10, 0)
        
        # Clinical Information
        st.header("Clinical Information")
        
        # Blood Test CSV Upload
        st.subheader("Blood Test Results")
        blood_test_file = st.file_uploader(
            "Upload Blood Test Results (TXT)", 
            type=['txt'],
            help="Upload a text file containing your blood test results"
        )
        
        # Preview blood test data if uploaded
        blood_test_data = None
        if blood_test_file is not None:
            blood_test_data = process_blood_test_file(blood_test_file)
            if blood_test_data:
                st.write("Blood Test Results Preview:")
                st.write(blood_test_data)
        
        er_status = st.radio("ER Status", ["Positive", "Negative", "Not Tested"])
        pr_status = st.radio("PR Status", ["Positive", "Negative", "Not Tested"])
        her2_status = st.radio("HER2 Status", ["Positive", "Negative", "Not Tested"])
        
        # Additional Notes
        notes = st.text_area("Additional Notes or Concerns")
        
        # Submit button
        submit_button = st.form_submit_button("Submit Monthly Update")
        
        if submit_button:
            if not visit_date or not provider_name or not facility_name:
                st.error("Please fill in all required fields")
                return None
            
            # Store the data
            monthly_data = {
                "visit_date": str(visit_date),
                "provider_name": provider_name,
                "facility_name": facility_name,
                "pain_level": pain_level,
                "fatigue_level": fatigue_level,
                "er_status": er_status,
                "pr_status": pr_status,
                "her2_status": her2_status,
                "notes": notes,
                "submission_date": datetime.now().isoformat()
            }
            
            # Add blood test data if available
            if blood_test_data:
                monthly_data["blood_test_results"] = blood_test_data
            
            # Initialize monthly_submissions in session state if it doesn't exist
            if 'monthly_submissions' not in st.session_state:
                st.session_state.monthly_submissions = []
            
            # Add new submission to the list
            st.session_state.monthly_submissions.append(monthly_data)
            
            # Update last submission date
            st.session_state.last_monthly_submission = str(visit_date)
            
            st.success("Monthly tracking information saved successfully!")
            return monthly_data
            
        return None

def is_monthly_submission_due():
    """
    Checks if it's time for a new monthly submission.
    Returns True if more than 30 days have passed since last submission.
    """
    if 'last_monthly_submission' not in st.session_state:
        return True
        
    last_submission = datetime.strptime(st.session_state.last_monthly_submission, '%Y-%m-%d')
    today = datetime.now()
    days_since_last = (today - last_submission).days
    
    return days_since_last >= 30