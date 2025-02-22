import streamlit as st

def main():
    st.title("Breast Cancer Risk Assessment")
    st.write("Upload your blood test (.txt) file and answer the following questions.")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload your blood test (.txt) file", type=["txt"])
    
    # User Input Questions
    age = st.number_input("Enter your age", min_value=18, max_value=120, step=1)
    family_history = st.radio("Do you have a family history of breast cancer?", ["Yes", "No"])
    smoking = st.radio("Do you smoke?", ["Yes", "No"])
    alcohol = st.radio("Do you consume alcohol regularly?", ["Yes", "No"])
    hormone_therapy = st.radio("Have you undergone hormone replacement therapy?", ["Yes", "No"])
    
    # Submit button
    if st.button("Analyze Data"):
        if uploaded_file is not None:
            st.success("File uploaded successfully! Processing data...")
            # Placeholder for ML model integration
            st.write("ML Model will process this data and generate a dashboard.")
        else:
            st.error("Please upload a .txt file to proceed.")

if __name__ == "__main__":
    main()