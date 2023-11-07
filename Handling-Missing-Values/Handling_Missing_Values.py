# import streamlit as st
# import pandas as pd
# import numpy as np

# st.title("Handling Missing Values")
# st.markdown(
#     f'''This Application is created and maintained by <a href="https://github.com/satheesh22g">*satheesh22g*</a>''',
#      unsafe_allow_html=True)
# data = st.file_uploader("Upload a Dataset", type=["csv"])
# if data is not None:
#     df = pd.read_csv(data)
#     st.dataframe(df.head())
#     st.success("Data Frame Loaded successfully")

#     if st.checkbox("Show dtypes with columns"):
#         st.text(df.dtypes)
#     if st.checkbox("Numerical Variables"):
#         n_cols = [x for x in df.select_dtypes(exclude=['object'])]
#         st.dataframe(df[n_cols].head(5))
#     if st.checkbox("Categorical Variables"):
#         c_cols = [x for x in df.select_dtypes(include=['object'])]
#         st.dataframe(df[c_cols].head(5))
#     if st.checkbox("Show Missing"):
#         st.write(df.isna().sum())
#     if st.checkbox("Show Missing Percentage"):
#         st.write((df.isnull().sum()/len(df))*100)

#     if st.checkbox("Drop Missing Values"):

#         # Create a selectbox widget
#         selected_option = st.selectbox("Select an option:", ["Drop Column with Null Values", "Drop Rows with Null Values"])
#         if st.checkbox("Drop All Values"):

#             st.write("You selected:", selected_option)
#         options_with_ids = {
#             "Option 1": 1,
#             "Option 2": 2,
#             "Option 3": 3
#         }

#         # Create a selectbox widget
#         selected_option_label = st.selectbox("Select an option:", list(options_with_ids.keys()))

#         # Get the selected option ID based on the selected label
#         selected_option_id = options_with_ids[selected_option_label]

#         # Display the selected option and its ID
#         st.write("You selected:", selected_option_label)
#         st.write("Option ID:", selected_option_id)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# App Title
st.title("Missing Values Handling App")

# Define a global variable for the DataFrame
df = None

# Upload Data
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df)

# Sidebar Controls
include_continuous_imputation = st.sidebar.checkbox("Mean/Median/Mode Imputation")
if include_continuous_imputation:
    mean_median_option = st.sidebar.radio("Choose Imputation for Continuous Columns:", ["Mean", "Median"])
include_deletion = st.sidebar.checkbox("Deletion")
include_mode = st.sidebar.checkbox("Mode Imputation for Categorical")
include_regression = st.sidebar.checkbox("Regression Imputation for Continuous")

# Data Handling Logic
if st.button("Handle Missing Data"):
    if df is not None:  # Check if df is not None
        st.write("Missing Value Counts Before Handling:")
        missing_values_before = df.isnull().sum()
        st.write(missing_values_before)

        df_copy = df.copy()  # Create a copy of the DataFrame

        if include_deletion:
            # Implement Deletion for all columns
            df_copy.dropna(inplace=True)

        if include_continuous_imputation:
            if mean_median_option == "Mean":
                # Implement Mean Imputation for continuous columns
                for col in df_copy.columns:
                    if df_copy[col].dtype != 'object':
                        imputer = SimpleImputer(strategy="mean")
                        df_copy[col] = imputer.fit_transform(df_copy[col].values.reshape(-1, 1))
            elif mean_median_option == "Median":
                # Implement Median Imputation for continuous columns
                for col in df_copy.columns:
                    if df_copy[col].dtype != 'object':
                        imputer = SimpleImputer(strategy="median")
                        df_copy[col] = imputer.fit_transform(df_copy[col].values.reshape(-1, 1))
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    imputer = SimpleImputer(strategy="most_frequent")
                    df_copy[col] = imputer.fit_transform(df_copy[col].values.reshape(-1, 1))

        if include_regression:
            numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_copy[col].isnull().any():
                    df_missing = df_copy[df_copy[col].isnull()]
                    df_not_missing = df_copy[~df_copy[col].isnull()]

                    numeric_columns_for_regression = df_not_missing.select_dtypes(include=[np.number]).columns

                    X_train = df_not_missing[numeric_columns_for_regression].drop(columns=[col])
                    y_train = df_not_missing[col]
                    X_test = df_missing[numeric_columns_for_regression].drop(columns=[col])

                    if len(X_train.columns) > 0:  # Check if there are numeric columns
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        df_copy.loc[df_copy[col].isnull(), col] = predictions

        st.write(f"Missing Value Counts After {method} Handling:")
        missing_values_after = df_copy.isnull().sum()
        st.write(missing_values_after)

    st.write("Handling Results:")
    st.write(df_copy)

