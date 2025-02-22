import pandas as pd
import re
from datetime import datetime

def validate_patient_data(data):
    """
    Validates the initial patient information form data.
    Returns True if data is valid, False otherwise.
    """
    required_fields = [
        'name', 'dob', 'gender', 'email', 'phone',
        'diagnosis_date', 'diagnosis_type'
    ]
    
    # Check required fields
    for field in required_fields:
        if not data.get(field):
            return False
    
    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, data['email']):
        return False
    
    # Validate phone number (basic check)
    phone_pattern = r'^\+?1?\d{9,15}$'
    if not re.match(phone_pattern, data['phone'].replace('-', '').replace(' ', '')):
        return False
    
    # Validate dates
    try:
        dob = datetime.strptime(data['dob'], '%Y-%m-%d')
        diagnosis_date = datetime.strptime(data['diagnosis_date'], '%Y-%m-%d')
        
        # Check if dates are not in future
        current_date = datetime.now()
        if dob > current_date or diagnosis_date > current_date:
            return False
            
        # Check if diagnosis date is after dob
        if diagnosis_date < dob:
            return False
    except ValueError:
        return False
    
    return True

def validate_monthly_data(data):
    """
    Validates the monthly tracking form data.
    Returns True if data is valid, False otherwise.
    """
    required_fields = [
        'visit_date', 'visit_type', 'provider_name',
        'facility_name'
    ]
    
    # Check required fields
    for field in required_fields:
        if not data.get(field):
            return False
    
    # Validate visit date
    try:
        visit_date = datetime.strptime(data['visit_date'], '%Y-%m-%d')
        if visit_date > datetime.now():
            return False
    except ValueError:
        return False
    
    # Validate numeric fields
    numeric_fields = ['pain_level', 'fatigue_level', 'stress_level', 'sleep_hours', 'ki67_value']
    for field in numeric_fields:
        if field in data:
            try:
                value = float(data[field])
                # Check ranges
                if field in ['pain_level', 'fatigue_level', 'stress_level'] and (value < 0 or value > 10):
                    return False
                elif field == 'sleep_hours' and (value < 0 or value > 24):
                    return False
                elif field == 'ki67_value' and (value < 0 or value > 100):
                    return False
            except (ValueError, TypeError):
                if data[field] is not None:  # Allow None values
                    return False
    
    return True

def parse_bloodtest(file):
    """
    Parses blood test results file and extracts relevant biomarkers.
    Returns a dictionary of biomarker values.
    """
    try:
        content = file.read().decode('utf-8')
        
        # Initialize biomarkers dictionary
        biomarkers = {
            'ER': None,
            'PR': None,
            'HER2': None,
            'Ki67': None,
            'CA153': None,
            'EGFR': None,
            'BRCA1': None,
            'BRCA2': None,
            'TP53': None
        }
        
        # Process each line
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                # Handle different value formats
                if any(status in value.lower() for status in ['positive', 'negative']):
                    biomarkers[key] = 'Positive' if 'positive' in value.lower() else 'Negative'
                else:
                    # Try to extract numeric value
                    match = re.search(r'[-+]?\d*\.?\d+', value)
                    if match:
                        number = float(match.group())
                        # Convert percentage to decimal if needed
                        if '%' in value:
                            number /= 100
                        biomarkers[key] = number
        
        return biomarkers
        
    except Exception as e:
        st.error(f"Error parsing blood test file: {str(e)}")
        return None

def format_report_data(monthly_data, patient_info):
    """
    Formats monthly tracking data for PDF report generation.
    Returns a dictionary of formatted data.
    """
    report_data = {
        'patient_name': patient_info['name'],
        'visit_date': datetime.strptime(monthly_data['visit_date'], '%Y-%m-%d').strftime('%B %d, %Y'),
        'provider': monthly_data['provider_name'],
        'facility': monthly_data['facility_name'],
        'symptoms_summary': {
            'pain_level': monthly_data['pain_level'],
            'fatigue_level': monthly_data['fatigue_level'],
            'reported_symptoms': ', '.join(monthly_data['symptoms']) if monthly_data['symptoms'] else 'None reported'
        },
        'biomarkers': {
            'ER': monthly_data['er_status'],
            'PR': monthly_data['pr_status'],
            'HER2': monthly_data['her2_status'],
            'Ki67': f"{monthly_data['ki67_value']}%"
        },
        'treatments': ', '.join(monthly_data['current_treatments']) if monthly_data['current_treatments'] else 'None reported',
        'lifestyle': {
            'exercise': monthly_data['exercise_changes'],
            'stress_level': monthly_data['stress_level'],
            'sleep_hours': monthly_data['sleep_hours']
        }
    }
    
    return report_data