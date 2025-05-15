import streamlit as st
import pandas as pd

st.title("Fraud Data Viewer")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of the Data", df.head())

    st.write("### Column Summary")
    st.write(df.describe(include='all'))

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Value Counts for 'is_fraud'")
    if 'is_fraud' in df.columns:
        st.write(df['is_fraud'].value_counts())
    else:
        st.warning("Column 'is_fraud' not found in the dataset.")

    st.write("### Column Data Types")
    st.write(df.dtypes)

    st.write("### Full Dataset (if needed)")
    if st.checkbox("Show full data"):
        st.dataframe(df)
