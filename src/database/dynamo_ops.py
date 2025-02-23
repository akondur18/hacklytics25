# src/database/dynamo_ops.py

import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import streamlit as st
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize DynamoDB resource
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
    region_name='us-east-1'
)

def save_patient_info(patient_data):
    """
    Saves initial patient information to DynamoDB.
    """
    try:
        table = dynamodb.Table('brca_patients')
        
        # Generate unique patient ID
        patient_id = str(uuid.uuid4())
        
        # Add metadata
        patient_data['patient_id'] = patient_id
        patient_data['created_at'] = datetime.now().isoformat()
        patient_data['updated_at'] = datetime.now().isoformat()
        patient_data['record_type'] = 'patient_info'
        
        # Save to DynamoDB
        table.put_item(Item=patient_data)
        
        # Store patient_id in session state
        st.session_state.patient_id = patient_id
        
        return patient_id
        
    except ClientError as e:
        st.error(f"Database error: {str(e)}")
        raise

def save_monthly_tracking(monthly_data):
    """
    Saves monthly tracking data to DynamoDB.
    """
    try:
        table = dynamodb.Table('brca_monthly_tracking')
        
        # Generate unique tracking ID
        tracking_id = str(uuid.uuid4())
        
        # Add metadata
        monthly_data['tracking_id'] = tracking_id
        monthly_data['created_at'] = datetime.now().isoformat()
        monthly_data['record_type'] = 'monthly_tracking'
        
        # Save to DynamoDB
        table.put_item(Item=monthly_data)
        
        return tracking_id
        
    except ClientError as e:
        st.error(f"Database error: {str(e)}")
        raise

def get_patient_info(patient_id):
    """
    Retrieves patient information from DynamoDB.
    """
    try:
        table = dynamodb.Table('brca_patients')
        response = table.get_item(Key={'patient_id': patient_id})
        
        if 'Item' in response:
            return response['Item']
        return None
        
    except ClientError as e:
        st.error(f"Database error: {str(e)}")
        raise

def get_patient_monthly_records(patient_id):
    """
    Retrieves all monthly tracking records for a patient.
    """
    try:
        table = dynamodb.Table('brca_monthly_tracking')
        
        # Query for all records with matching patient_id
        response = table.query(
            IndexName='patient_id-visit_date-index',
            KeyConditionExpression='patient_id = :pid',
            ExpressionAttributeValues={':pid': patient_id}
        )
        
        return response.get('Items', [])
        
    except ClientError as e:
        st.error(f"Database error: {str(e)}")
        raise

def verify_user(email, password):
    """
    Verify user credentials
    """
    try:
        table = dynamodb.Table('brca_users')
        response = table.get_item(Key={'email': email})
        
        if 'Item' in response:
            stored_password = response['Item']['password']
            return stored_password == hash_password(password)
        return False
        
    except ClientError as e:
        st.error(f"Authentication error: {str(e)}")
        return False

def create_user(email, password):
    """
    Create a new user
    """
    try:
        table = dynamodb.Table('brca_users')
        
        # Check if user already exists
        response = table.get_item(Key={'email': email})
        if 'Item' in response:
            return False
            
        # Create new user
        table.put_item(
            Item={
                'email': email,
                'password': hash_password(password),
                'created_at': datetime.now().isoformat()
            }
        )
        return True
        
    except ClientError as e:
        st.error(f"Registration error: {str(e)}")
        return False

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(str.encode(password)).hexdigest()