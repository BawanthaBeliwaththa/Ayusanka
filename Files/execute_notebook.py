import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def execute():
    notebook_filename = "Sri_Lanka_Life_Expectancy_Model.ipynb"
    print(f"Reading notebook: {notebook_filename}...")
    try:
        with open(notebook_filename, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
            
        print("Executing notebook code cells. This will render the charts, heatmaps, and summary tables...")
        # Configure the execution preprocessor
        ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
        
        # Run the notebook
        ep.preprocess(nb, {"metadata": {"path": "."}})
        
        print(f"Saving executed notebook back to: {notebook_filename}...")
        with open(notebook_filename, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
            
        print("Execution successful! The notebook now has pre-rendered visualizations embedded.")
        
    except Exception as e:
        print("Error during notebook execution:", e)
        sys.exit(1)

if __name__ == "__main__":
    execute()
