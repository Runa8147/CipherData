import streamlit as st
import requests
import json
import pandas as pd
import io

API_KEY = st.secrets['GEMINI_API_KEY']
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

MODELS = {
    "gemini-pro": f"{BASE_URL}/gemini-pro:generateContent",
    "gemini-pro-vision": f"{BASE_URL}/gemini-pro-vision:generateContent",
    "embedding-001": f"{BASE_URL}/embedding-001:embedContent",
    # Add other models as they become available
}

def generate_synthetic_data(schema, num_rows, model):
    prompt = f"""
    Generate a synthetic dataset with {num_rows} rows based on the following schema:
    {json.dumps(schema, indent=2)}
    
    The data should be realistic but entirely fictional. 
    Provide the data in CSV format without headers.
    """
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }
    data = {
        "contents": [{"parts":[{"text": prompt}]}]
    }
    
    response = requests.post(MODELS[model], headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code}, {response.text}"

def main():
    st.title("Synthetic Data Generator")
    
    # Model selection
    model = st.selectbox("Select Gemini Model", list(MODELS.keys()))
    
    # User input for schema
    st.subheader("Define Your Data Schema")
    num_fields = st.number_input("Number of fields", min_value=1, max_value=10, value=3)
    
    schema = {}
    for i in range(num_fields):
        col1, col2 = st.columns(2)
        with col1:
            field_name = st.text_input(f"Field {i+1} Name", key=f"name_{i}")
        with col2:
            field_type = st.selectbox(f"Field {i+1} Type", ["string", "integer", "float", "date", "boolean"], key=f"type_{i}")
        schema[field_name] = field_type
    
    # Number of rows to generate
    num_rows = st.number_input("Number of rows to generate", min_value=1, max_value=100, value=10)
    
    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating data..."):
            csv_data = generate_synthetic_data(schema, num_rows, model)
        
        # Convert CSV string to DataFrame
        df = pd.read_csv(io.StringIO(csv_data), header=None, names=schema.keys())
        
        # Display the data
        st.subheader("Generated Synthetic Data")
        st.dataframe(df)
        
        # Provide download link
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="synthetic_data.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()