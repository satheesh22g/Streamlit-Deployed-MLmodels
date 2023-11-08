import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to replace outliers with mean value using IQR
def replace_outliers_with_mean_iqr(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
    data[column_name] = data[column_name].apply(lambda x: x if (x >= lower_bound and x <= upper_bound) else data[column_name].mean())
    return data, outliers

# Function to replace outliers with mean value using Z-Score
def replace_outliers_with_mean_zscore(data, column_name):
    z_scores = (data[column_name] - data[column_name].mean()) / data[column_name].std()
    outliers = data[abs(z_scores) > 3]
    data[column_name] = data.apply(lambda row: row[column_name] if abs(z_scores[row.name]) <= 3 else data[column_name].mean(), axis=1)
    return data, outliers

st.title("Outlier Handling App")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.header("Original Dataset")

    if st.button("Show Dataset"):
        st.write(data)

    # Detect and handle outliers
    for column in data.columns:
        if data[column].dtype == 'int64' or data[column].dtype == 'float64':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            
            if not outliers.empty:
                st.subheader(f"Outliers in Column: {column}")
                fig, ax = plt.subplots()
                ax.boxplot(data[column])
                
                if st.button(f"Show Box Plot for {column}"):
                    st.pyplot(fig)
                
                method = st.selectbox(f"Select Outlier Handling Method for {column}", ("Replace with Mean", "Delete Outliers"))
                if method == "Replace with Mean" and st.button(f"Replace Outliers for {column}"):
                    data, replaced_outliers = replace_outliers_with_mean_iqr(data, column)
                    st.write(f"Replaced outliers in {column} with mean value. Outliers: {replaced_outliers.to_string(index=False)}")
                elif method == "Delete Outliers" and st.button(f"Delete Outliers for {column}"):
                    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
                    st.write(f"Deleted outliers in {column}. Outliers: {outliers.to_string(index=False)}")

    if st.button("Show End Results"):
        st.header("Processed Dataset")
        st.write(data)
