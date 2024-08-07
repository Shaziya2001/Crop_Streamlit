import streamlit as st
import nbformat
from nbconvert import PythonExporter
from io import StringIO
import contextlib

def run_notebook(notebook_path, selected_features):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
        
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(notebook)
    
    # Prepare the feature list and replace placeholders
    feature_list_str = ", ".join([f"'{feature}'" for feature in selected_features])
    source = source.replace("selected_features_placeholder", feature_list_str)
    
    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(source, globals())
    
    return buffer.getvalue()

st.title("Jupyter Notebook Viewer with Feature Selection")

# Example feature list (replace with actual feature names from your dataset)
feature_list = ['time_idx', 'Rain', 'Pesticide', 'Fertilizer', 'Crop Yield']

# User input widget for feature selection
selected_features = st.multiselect("Select Features", options=feature_list, default=feature_list[:2])

notebook_path = 'TFT.ipynb'  # Path to the notebook file in the GitHub repo
if st.button("Run Notebook"):
    try:
        notebook_output = run_notebook(notebook_path, selected_features)
        st.text(notebook_output)
    except Exception as e:
        st.error(f"Error running notebook: {e}")


