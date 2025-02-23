import streamlit as st
from datetime import datetime
import boto3
import pandas as pd
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

def main():
    st.set_page_config(
        page_title="BRCA Care Portal",
        page_icon="🩺",
        layout="wide"
    )

    # Check authentication
    if not st.session_state.authenticated:
        render_auth_page()
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Initial Form", "Monthly Tracking", "View History"],
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

    elif page == "View History":
        if not st.session_state.initial_form_submitted:
            st.warning("Please complete the initial form first.")
        else:
            display_patient_history()

if __name__ == "__main__":
    main()