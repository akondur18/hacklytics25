import streamlit as st
import pandas as pd
from datetime import datetime
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'monthly_submissions' not in st.session_state:
    st.session_state.monthly_submissions = []

def fetch_mock_biomarker_data():
    """Return mock biomarker data for demonstration"""
    return [
        {"date": "2023-10-01", "ER": 1.2, "PR": 0.8, "HER2": 15.5, "KI67": 0.25},
        {"date": "2023-09-01", "ER": 1.1, "PR": 0.7, "HER2": 14.8, "KI67": 0.22},
        {"date": "2023-08-01", "ER": 1.0, "PR": 0.6, "HER2": 14.0, "KI67": 0.20},
    ]

def plot_biomarker_trends(data):
    """Plot biomarker trends using both matplotlib and plotly"""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create Plotly figure for interactive visualization
    fig = go.Figure()
    for column in ['ER', 'PR', 'HER2', 'KI67']:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[column],
            name=column,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Biomarker Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Biomarker Levels",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def display_key_metrics(data):
    """Display latest biomarker metrics"""
    latest_data = data[-1]
    
    st.subheader("Latest Biomarker Levels")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ER Status", f"{latest_data['ER']:.2f} fmol/mL")
    with col2:
        st.metric("PR Status", f"{latest_data['PR']:.2f} fmol/mL")
    with col3:
        st.metric("HER2 Status", f"{latest_data['HER2']:.2f} ng/mL")
    with col4:
        st.metric("Ki-67 Index", f"{latest_data['KI67'] * 100:.1f}%")

def display_patient_history():
    """Display patient history and tracking information"""
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

    if monthly_submissions:
        df = pd.DataFrame(monthly_submissions)
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        df = df.sort_values('visit_date')
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Pain & Fatigue Tracking", 
            "Biomarker Status", 
            "Visit History", 
            "Export Data"
        ])
        
        with tab1:
            st.subheader("Pain and Fatigue Levels Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['visit_date'], y=df['pain_level'],
                                   name='Pain Level', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=df['visit_date'], y=df['fatigue_level'],
                                   name='Fatigue Level', mode='lines+markers'))
            fig.update_layout(
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
            visit_history = df[['visit_date', 'provider_name', 'facility_name', 'notes']]
            for _, row in visit_history.iterrows():
                with st.expander(f"Visit on {row['visit_date'].strftime('%Y-%m-%d')}"):
                    st.write(f"Provider: {row['provider_name']}")
                    st.write(f"Facility: {row['facility_name']}")
                    if row['notes']:
                        st.write("Notes:")
                        st.write(row['notes'])
                        
        with tab4:
            st.subheader("Export Patient Data")
            
            # Date range selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    min(df['visit_date']).date() if not df.empty else datetime.now().date()
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    max(df['visit_date']).date() if not df.empty else datetime.now().date()
                )
            
            # Data selection
            st.subheader("Select Data to Export")
            export_options = st.multiselect(
                "Choose data to include:",
                options=["Patient Information", "Biomarker Data", "Visit History", "Pain & Fatigue Tracking"],
                default=["Patient Information", "Biomarker Data", "Visit History", "Pain & Fatigue Tracking"]
            )
            
            # Filter data based on date range
            mask = (df['visit_date'].dt.date >= start_date) & (df['visit_date'].dt.date <= end_date)
            filtered_df = df.loc[mask]
            
            # Create different dataframes based on selection
            export_dfs = {}
            if "Patient Information" in export_options:
                export_dfs["patient_info"] = pd.DataFrame([patient_info])
            if "Biomarker Data" in export_options:
                export_dfs["biomarker_data"] = filtered_df[['visit_date', 'er_status', 'pr_status', 'her2_status']]
            if "Visit History" in export_options:
                export_dfs["visit_history"] = filtered_df[['visit_date', 'provider_name', 'facility_name', 'notes']]
            if "Pain & Fatigue Tracking" in export_options:
                export_dfs["pain_fatigue"] = filtered_df[['visit_date', 'pain_level', 'fatigue_level']]
            
            # Preview section
            st.subheader("Data Preview")
            for name, df_preview in export_dfs.items():
                with st.expander(f"Preview {name}"):
                    st.dataframe(df_preview)
            
            # Export format selection
            export_format = st.selectbox(
                "Select export format:",
                ["CSV", "Excel", "PDF Report"]
            )
            
            if st.button("Generate Export"):
                if export_format == "CSV":
                    # Export each selected dataset as a separate CSV
                    for name, df_export in export_dfs.items():
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label=f"Download {name} (CSV)",
                            data=csv,
                            file_name=f"{name}_{start_date}_{end_date}.csv",
                            mime="text/csv",
                            key=f"csv_{name}"
                        )
                
                elif export_format == "Excel":
                    # Create Excel file with multiple sheets
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer) as writer:
                        for name, df_export in export_dfs.items():
                            df_export.to_excel(writer, sheet_name=name, index=False)
                    
                    st.download_button(
                        label="Download Excel File",
                        data=buffer.getvalue(),
                        file_name=f"patient_data_{start_date}_{end_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                else:  # PDF Report
                    st.info("Generating PDF report... (This feature would require additional PDF generation library)")
                    # Here you could add actual PDF generation using libraries like ReportLab or PyFPDF
            
           

def main():
    st.title("Comprehensive Patient Dashboard 🏥")
    
    # Fetch biomarker data
    biomarker_data = fetch_mock_biomarker_data()
    
    # Create tabs for main dashboard sections
    tab1, tab2 = st.tabs(["Biomarker Analysis", "Patient History"])
    
    with tab1:
        if biomarker_data:
            display_key_metrics(biomarker_data)
            plot_biomarker_trends(biomarker_data)
        else:
            st.warning("No biomarker data available.")
            
    with tab2:
        display_patient_history()

if __name__ == "__main__":
    main()