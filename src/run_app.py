
import subprocess
from histology_adapter import parse_optimize_output

def run_optimized_app():

    histology = parse_optimize_output()
    

    subprocess.run([
        'streamlit', 'run', 'main.py', 
        '--',  # Pass arguments after this
        '--cancer_type', histology
    ])

if __name__ == '__main__':
    run_optimized_app()