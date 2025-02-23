import streamlit as st
import pandas as pd
from datetime import datetime
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'monthly_submissions' not in st.session_state:
    st.session_state.monthly_submissions = []
if 'blood_test_data' not in st.session_state:
    st.session_state.blood_test_data = []

def parse_blood_test_file(content):
    """Parse blood test text content into structured data"""
    lines = content.split('\n')
    data = {}
    
    # Extract date using regex
    date_match = re.search(r'Date: (\d{4}-\d{2}-\d{2})', content)
    if date_match:
        data['date'] = date_match.group(1)
    
    # Extract values for different categories
    current_category = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.endswith(':'):
            current_category = line[:-1]
            continue
            
        if line.startswith('-'):
            # Remove the dash and split by colon
            parts = line[1:].strip().split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                # Extract numeric value using regex
                value_match = re.search(r'(\d+\.?\d*)', parts[1])
                if value_match:
                    data[key] = float(value_match.group(1))
    
    return data

def display_blood_test_analysis():
    """Display blood test analysis dashboard"""
    st.header("Blood Test Analysis 🔬")
    
    # Upload blood test files
    uploaded_files = st.file_uploader(
        "Upload Blood Test Results", 
        type=['txt'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Process all uploaded files
        all_data = []
        for file in uploaded_files:
            content = file.getvalue().decode()
            parsed_data = parse_blood_test_file(content)
            if parsed_data:
                all_data.append(parsed_data)
        
        # Store in session state
        st.session_state.blood_test_data = all_data
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Tumor Markers", 
            "Cell Fragments", 
            "Organ Function",
            "Trend Analysis"
        ])
        
        with tab1:
            st.subheader("Tumor Markers Trends")
            
            # Create subplot for tumor markers
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("HER2", "CA 15-3", "CA 27-29", "Ki-67")
            )
            
            markers = ["HER2", "CA 15-3", "CA 27-29", "Ki-67"]
            positions = [(1,1), (1,2), (2,1), (2,2)]
            
            for marker, pos in zip(markers, positions):
                if marker in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['date'], y=df[marker], name=marker),
                        row=pos[0], col=pos[1]
                    )
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add metrics for latest values
            if not df.empty:
                st.subheader("Latest Values")
                latest = df.iloc[-1]
                cols = st.columns(4)
                for col, marker in zip(cols, markers):
                    if marker in latest:
                        with col:
                            st.metric(marker, f"{latest[marker]:.1f}")
        
        with tab2:
            st.subheader("Cell Fragment Analysis")
            if 'Circulating Tumor DNA (ctDNA)' in df.columns:
                timeline_data = []
                for _, row in df.iterrows():
                    timeline_data.append({
                        'date': row['date'],
                        'ctDNA': 'Detected' in str(row.get('Circulating Tumor DNA (ctDNA)', '')),
                        'CTCs': 'Detected' in str(row.get('Circulating Tumor Cells (CTCs)', ''))
                    })
                
                timeline_df = pd.DataFrame(timeline_data)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timeline_df['date'],
                    y=timeline_df['ctDNA'].astype(int),
                    name='ctDNA',
                    mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    x=timeline_df['date'],
                    y=timeline_df['CTCs'].astype(int),
                    name='CTCs',
                    mode='lines+markers'
                ))
                
                fig.update_layout(
                    title="Cell Fragment Detection Timeline",
                    yaxis=dict(
                        ticktext=["Not Detected", "Detected"],
                        tickvals=[0, 1]
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Organ Function Parameters")
            
            # Create subplot for organ function tests
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Kidney Function", "Liver Function", "Enzyme Levels", "Alkaline Phosphatase")
            )
            
            # Kidney function
            if 'BUN' in df.columns and 'Creatinine' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['date'], y=df['BUN'], name="BUN"),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['date'], y=df['Creatinine'], name="Creatinine"),
                    row=1, col=1
                )
            
            # Liver function
            if 'ALT' in df.columns and 'AST' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['date'], y=df['ALT'], name="ALT"),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=df['date'], y=df['AST'], name="AST"),
                    row=1, col=2
                )
            
            # Alkaline Phosphatase
            if 'Alkaline Phosphatase' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['date'], y=df['Alkaline Phosphatase'], 
                              name="Alkaline Phosphatase"),
                    row=2, col=1
                )
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Trend Analysis and Insights")
            
            # Calculate percentage changes for key markers
            markers_to_track = ['HER2', 'CA 15-3', 'CA 27-29', 'Ki-67']
            cols = st.columns(len(markers_to_track))
            
            for col, marker in zip(cols, markers_to_track):
                if marker in df.columns:
                    first_value = df[marker].iloc[0]
                    last_value = df[marker].iloc[-1]
                    change = ((last_value - first_value) / first_value) * 100
                    
                    with col:
                        st.metric(
                            f"{marker} Change",
                            f"{last_value:.1f}",
                            f"{change:.1f}%"
                        )
            
            # Create correlation heatmap
            numeric_cols = [col for col in ['HER2', 'CA 15-3', 'CA 27-29', 'Ki-67', 
                                          'BUN', 'Creatinine', 'ALT', 'AST', 
                                          'Alkaline Phosphatase'] if col in df.columns]
            
            if numeric_cols:
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=numeric_cols,
                    y=numeric_cols
                )
                fig.update_layout(title="Biomarker Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Add summary statistics
                st.subheader("Summary Statistics")
                st.dataframe(df[numeric_cols].describe())

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
            export_options = st.multiselect(
                "Choose data to include:",
                options=["Patient Information", "Blood Test Data", "Visit History", "Pain & Fatigue Tracking"],
                default=["Patient Information", "Blood Test Data", "Visit History", "Pain & Fatigue Tracking"]
            )
            
            # Filter data based on date range
            mask = (df['visit_date'].dt.date >= start_date) & (df['visit_date'].dt.date <= end_date)
            filtered_df = df.loc[mask]
            
            # Create different dataframes based on selection
            export_dfs = {}
            if "Patient Information" in export_options:
                export_dfs["patient_info"] = pd.DataFrame([patient_info])
            if "Blood Test Data" in export_options and st.session_state.blood_test_data:
                export_dfs["blood_test_data"] = pd.DataFrame(st.session_state.blood_test_data)
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

def fetch_mock_biomarker_data():
    """Return mock biomarker data for demonstration"""
    return [
        {"date": "2023-10-01", "ER": 1.2, "PR": 0.8, "HER2": 15.5, "KI67": 0.25},
        {"date": "2023-09-01", "ER": 1.1, "PR": 0.7, "HER2": 14.8, "KI67": 0.22},
        {"date": "2023-08-01", "ER": 1.0, "PR": 0.6, "HER2": 14.0, "KI67": 0.20},
    ]

def main():
    st.title("Comprehensive Patient Dashboard 🏥")
    
    # Create tabs for main dashboard sections
    tab1, tab2, tab3 = st.tabs(["Blood Test Analysis", "Biomarker Analysis", "Patient History"])
    
    with tab1:
        display_blood_test_analysis()
        
    with tab2:
        # Fetch biomarker data
        biomarker_data = fetch_mock_biomarker_data()
        if biomarker_data:
            display_key_metrics(biomarker_data)
            plot_biomarker_trends(biomarker_data)
        else:
            st.warning("No biomarker data available.")
            
    with tab3:
        display_patient_history()

if __name__ == "__main__":
    main()