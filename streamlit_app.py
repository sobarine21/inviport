import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import io
import plotly.express as px

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="AI-Powered Portfolio Reconciliation", layout="wide")
st.title("AI-Powered Automated Reporting, Aggregation & Reconciliation")
st.write(
    """
    Upload your portfolio holdings, broker trades, or raw fund data in Excel or CSV format. 
    The application dynamically adapts to any file structure, aggregates and reconciles data, 
    generates a downloadable report, and provides detailed visualizations.
    """
)

# --- File uploader (multiple file types and multiple files) ---
uploaded_files = st.file_uploader(
    "Upload portfolio holding, broker trades, and/or fund data files (Excel or CSV). You can upload multiple files.",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True
)

# --- Helper Functions ---
def read_file(file):
    """Read Excel or CSV file, return DataFrame, and original sheetname if Excel."""
    filename = file.name
    if filename.endswith('.csv'):
        df = pd.read_csv(file)
        return [(df, filename)]
    elif filename.endswith(('.xls', '.xlsx')):
        sheets = pd.read_excel(file, sheet_name=None)
        # Return all sheets as (DataFrame, sheet name)
        return [(sheets[sheet], f"{filename} - {sheet}") for sheet in sheets]
    else:
        return []

def summarize_dataframe(df):
    """Return some basic statistics and info about the DataFrame."""
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict()
    }
    return summary

def ai_analyze(files_info):
    """Use Gemini AI to analyze the uploaded dataframes and suggest/perform reconciliation."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    ai_results = []
    for df, name in files_info:
        sample_data = df.head(5).to_dict()
        prompt = f"""
        Analyze the following data from a file named '{name}'. 
        Suggest the key columns for portfolio, trades, or fund data, 
        and recommend aggregation/reconciliation logic. 
        Data sample: {sample_data}
        """
        try:
            response = model.generate_content(prompt)
            ai_results.append((name, response.text))
        except Exception as e:
            ai_results.append((name, f"AI Analysis Error: {e}"))
    return ai_results

def merge_dataframes(dfs):
    """Dynamically merge dataframes on best-guess join columns using AI and heuristics."""
    if len(dfs) <= 1:
        return dfs[0][0] if dfs else None, "Only one file provided. No merging performed."
    
    # Let AI suggest columns to join on
    model = genai.GenerativeModel('gemini-2.0-flash')
    join_suggestions = []
    all_samples = {name: df.head(5).to_dict() for df, name in dfs}
    prompt = f"""
    Given the following data samples from uploaded files: {all_samples},
    suggest the best columns to join the dataframes on for portfolio reconciliation.
    Only return a Python dict mapping file names to join column names.
    """
    try:
        response = model.generate_content(prompt)
        # Attempt to parse the AI's response as Python dict (safe eval)
        import ast
        join_dict = ast.literal_eval(response.text)
        join_cols = [join_dict.get(name, None) for _, name in dfs]
        # If join columns are not found for all, fallback to first common column
        for idx, (df, name) in enumerate(dfs):
            if not join_cols[idx]:
                commons = set(df.columns)
                for jdx, (df2, name2) in enumerate(dfs):
                    if idx != jdx:
                        commons = commons & set(df2.columns)
                join_cols[idx] = list(commons)[0] if commons else None
    except Exception as e:
        # Fallback: use intersection of columns
        join_cols = []
        commons = set(dfs[0][0].columns)
        for df, name in dfs[1:]:
            commons = commons & set(df.columns)
        join_on = list(commons)[0] if commons else None
        join_cols = [join_on for _ in dfs]

    # Try merging all dataframes on suggested columns
    merged = dfs[0][0]
    for i in range(1, len(dfs)):
        left_col = join_cols[0]
        right_col = join_cols[i]
        if left_col and right_col:
            merged = pd.merge(
                merged, dfs[i][0], left_on=left_col, right_on=right_col, how='outer', suffixes=('', f'_{i}')
            )
        else:
            merged = pd.concat([merged, dfs[i][0]], axis=0, ignore_index=True)
    return merged, f"Data merged using columns: {join_cols}"

def generate_visualizations(df):
    """Generate dynamic visualizations based on data types and columns."""
    st.subheader("Dynamic Data Visualizations")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) >= 1:
        for col in numeric_cols:
            st.write(f"Histogram for {col}:")
            fig = px.histogram(df, x=col, nbins=20, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
    if len(numeric_cols) >= 2:
        st.write("Scatter plot of first two numeric columns:")
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
        st.plotly_chart(fig, use_container_width=True)
    # Pie chart for categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        st.write(f"Pie chart for {cat_cols[0]} by {numeric_cols[0]}:")
        pie_data = df.groupby(cat_cols[0])[numeric_cols[0]].sum().reset_index()
        fig = px.pie(pie_data, names=cat_cols[0], values=numeric_cols[0])
        st.plotly_chart(fig, use_container_width=True)

# --- Main Logic ---
if uploaded_files:
    files_info = []
    for file in uploaded_files:
        files_info.extend(read_file(file))
    
    # Display summary of uploaded data
    st.subheader("Uploaded Data Files Preview & Summary")
    for df, name in files_info:
        st.markdown(f"**{name}**")
        st.dataframe(df.head(10))
        st.json(summarize_dataframe(df))

    # AI-driven analysis and suggestions
    st.subheader("AI-driven Data Analysis & Suggestions")
    ai_results = ai_analyze(files_info)
    for name, result in ai_results:
        with st.expander(f"AI Analysis for {name}"):
            st.write(result)

    # Merge & aggregate
    st.subheader("Data Aggregation & Reconciliation")
    merged_df, merge_info = merge_dataframes(files_info)
    st.write(merge_info)
    if merged_df is not None:
        st.dataframe(merged_df.head(20))

        # Downloadable output
        st.subheader("Download Aggregated & Reconciled Data")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            merged_df.to_excel(writer, index=False, sheet_name='Aggregated')
            for idx, (df, name) in enumerate(files_info):
                df.to_excel(writer, index=False, sheet_name=f"Source_{idx+1}")
            writer.save()
        st.download_button(
            label="Download Excel Report",
            data=output.getvalue(),
            file_name="aggregated_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Show dynamic visualizations
        generate_visualizations(merged_df)
    else:
        st.warning("Unable to merge data. Please check file compatibility.")
else:
    st.info("Please upload at least one Excel or CSV file to begin.")
