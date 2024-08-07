import nbformat
from nbconvert import PythonExporter
from io import StringIO
import contextlib
import streamlit as st
import sys

def run_notebook(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
        
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(notebook)
    
    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(source, globals())
    
    return buffer.getvalue()

notebook_path = 'https://github.com/Shaziya2001/Crop_Streamlit/blob/main/TFT.ipynb'  

st.title("Jupyter Notebook Viewer")

try:
    notebook_output = run_notebook(notebook_path)
    st.text(notebook_output)
except Exception as e:
    st.error(f"Error running notebook: {e}")
