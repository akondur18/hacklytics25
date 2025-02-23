# src/components/auth.py

import streamlit as st
import hashlib
from datetime import datetime
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
    region_name='us-east-1'
)

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def render_auth_page():
    st.title("BRCA Care Portal 🔒")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if verify_credentials(email, password):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register_form"):
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif register_user(new_email, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Registration failed. Email might already be registered.")

def verify_credentials(email, password):
    return True

def register_user(email, password):
    return True

# def verify_credentials(email, password):
#     """Verify user credentials."""
#     try:
#         table = dynamodb.Table('brca_users')
#         response = table.get_item(
#             Key={'email': email}
#         )
        
#         if 'Item' in response:
#             stored_password = response['Item']['password']
#             return stored_password == hash_password(password)
#         return False
        
#     except Exception as e:
#         st.error(f"Authentication error: {str(e)}")
#         return False

# def register_user(email, password):
#     """Register a new user."""
#     try:
#         table = dynamodb.Table('brca_users')
        
#         # Check if user already exists
#         response = table.get_item(
#             Key={'email': email}
#         )
#         if 'Item' in response:
#             return False
            
#         # Create new user
#         table.put_item(
#             Item={
#                 'email': email,
#                 'password': hash_password(password),
#                 'created_at': str(datetime.now())
#             }
#         )
#         return True
        
#     except Exception as e:
#         st.error(f"Registration error: {str(e)}")
#         return False