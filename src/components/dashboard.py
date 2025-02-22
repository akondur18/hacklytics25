# src/components/dashboard.py

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def display_patient_history():
    st.title("Patient History Dashboard 📊")
    
    # Get patient info and monthly submissions from session state
    patient_info = st.session_state.get('patient_info', {})
    monthly_submissions = st.session_state.get('monthly_submissions', [])
    
    # Display Patient Info
    st.header("Patient Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patient Name", patient_info.get('name', 'N/A'))
    with col2:
        st.metric("Diagnosis Type", patient_info.get('diagnosis_type', 'N/A'))
    with col3:
        st.metric("Last Update", 
                 monthly_submissions[-1]['visit_date'] if monthly_submissions else 'No records')

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Pain & Fatigue Tracking", "Biomarker Status", "Visit History"])
    
    if monthly_submissions:
        # Convert submissions to DataFrame
        df = pd.DataFrame(monthly_submissions)
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        df = df.sort_values('visit_date')
        
        with tab1:
            st.subheader("Pain and Fatigue Levels Over Time")
            
            # Create line chart for pain and fatigue
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['visit_date'], y=df['pain_level'],
                                   name='Pain Level', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=df['visit_date'], y=df['fatigue_level'],
                                   name='Fatigue Level', mode='lines+markers'))
            fig.update_layout(
                title="Pain and Fatigue Tracking",
                xaxis_title="Visit Date",
                yaxis_title="Level (0-10)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Biomarker Status History")
            
            # Create a table for biomarker status
            biomarker_df = df[['visit_date', 'er_status', 'pr_status', 'her2_status']]
            st.dataframe(biomarker_df)
            
            # Create pie charts for latest status
            latest = df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = px.pie(values=[1], names=[latest['er_status']],
                           title='Latest ER Status')
                st.plotly_chart(fig)
                
            with col2:
                fig = px.pie(values=[1], names=[latest['pr_status']],
                           title='Latest PR Status')
                st.plotly_chart(fig)
                
            with col3:
                fig = px.pie(values=[1], names=[latest['her2_status']],
                           title='Latest HER2 Status')
                st.plotly_chart(fig)
            
        with tab3:
            st.subheader("Visit History")
            
            # Create timeline of visits
            visit_history = df[['visit_date', 'provider_name', 'facility_name', 'notes']]
            for _, row in visit_history.iterrows():
                with st.expander(f"Visit on {row['visit_date'].strftime('%Y-%m-%d')}"):
                    st.write(f"Provider: {row['provider_name']}")
                    st.write(f"Facility: {row['facility_name']}")
                    if row['notes']:
                        st.write("Notes:")
                        st.write(row['notes'])
    else:
        st.info("No monthly tracking data available yet. Complete your first monthly submission to see your health trends.")

    # Add download functionality
    if monthly_submissions:
        st.sidebar.header("Export Data")
        if st.sidebar.button("Download History as CSV"):
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Click to Download",
                data=csv,
                file_name="patient_history.csv",
                mime="text/csv"
            )