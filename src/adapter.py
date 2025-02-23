# histology_adapter.py
import subprocess
import ast
import pandas as pd
from main import generate_treatment_plan  # Import from your existing main.py

def parse_optimize_output():
    """Capture and parse output from optimize_model.py"""
    result = subprocess.run(['python', 'optimize_model.py'], capture_output=True, text=True)
    
    # Extract predicted histology from sample predictions
    predictions_str = result.stdout.split("Sample Predictions vs Actual (Best Model):\n")[1].split("\n\n")[0]
    df = pd.read_csv(pd.compat.StringIO(predictions_str), sep="\s{2,}", engine="python")
    
    return df['Predicted'].iloc[0]  # Return first prediction

def generate_histology_plan(clinical_data):
    """Bridge function to connect optimize_model output with main.py"""
    predicted_histology = parse_optimize_output()
    
    # Call existing function from main.py with model-predicted histology
    return generate_treatment_plan(
        cancer_type=predicted_histology,
        clinical_data=clinical_data  # Should match your existing data structure
    )

# Example usage in your Streamlit app:
# In main.py's render_analysis_section(), replace:
# treatment_plan = generate_treatment_plan(cancer_type, clinical_data)
# With:
# from histology_adapter import generate_histology_plan
# treatment_plan = generate_histology_plan(clinical_data)