# dashboard.py
import streamlit as st
import pandas as pd
import boto3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# AWS Configuration (same as in main.py)
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

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Function to fetch data from DynamoDB (or any other source)
def fetch_data(user_email):
    # Example: Fetch data from DynamoDB
    # table = dynamodb.Table(DYNAMO_TABLE)
    # response = table.query(
    #     KeyConditionExpression='user_email = :email',
    #     ExpressionAttributeValues={':email': user_email}
    # )
    # return response['Items']
    
    # For now, return mock data
    mock_data = [
        {"date": "2023-10-01", "ER": 1.2, "PR": 0.8, "HER2": 15.5, "KI67": 0.25},
        {"date": "2023-09-01", "ER": 1.1, "PR": 0.7, "HER2": 14.8, "KI67": 0.22},
        {"date": "2023-08-01", "ER": 1.0, "PR": 0.6, "HER2": 14.0, "KI67": 0.20},
    ]
    return mock_data

# Function to plot biomarker trends
def plot_biomarker_trends(data):
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    st.subheader("Biomarker Trends Over Time")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, ax=ax)
    ax.set_ylabel("Biomarker Levels")
    ax.set_xlabel("Date")
    st.pyplot(fig)

# Function to display key metrics
def display_key_metrics(data):
    latest_data = data[-1]  # Get the latest data point
    
    st.subheader("Latest Biomarker Levels")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ER Status", f"{latest_data['ER']:.2f} fmol/mL")
    with col2:
        st.metric("PR Status", f"{latest_data['PR']:.2f} fmol/mL")
    with col3:
        st.metric("HER2 Status", f"{latest_data['HER2']:.2f} ng/mL")
    
    st.metric("Ki-67 Index", f"{latest_data['KI67'] * 100:.1f}%")

# Main Dashboard Function
def dashboard():
    st.title("Breast Cancer Biomarker Dashboard")
    
    # Fetch data for the logged-in user
    user_email = "user@example.com"  # Replace with actual user email from session
    data = fetch_data(user_email)
    
    if data:
        display_key_metrics(data)
        plot_biomarker_trends(data)
    else:
        st.warning("No data available for this user.")

# Run the dashboard
if __name__ == "__main__":
    dashboard()