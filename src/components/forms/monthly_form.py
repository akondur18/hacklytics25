# src/components/forms/monthly_form.py

import streamlit as st
from datetime import datetime

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
            
            # Initialize monthly_submissions in session state if it doesn't exist
            if 'monthly_submissions' not in st.session_state:
                st.session_state.monthly_submissions = []
            
            # Add new submission to the list
            st.session_state.monthly_submissions.append(monthly_data)
            
            # Update last submission date
            st.session_state.last_monthly_submission = str(visit_date)
            
            st.success("Monthly tracking information saved successfully!")
            return monthly_data
            
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
            
            # Update session state
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