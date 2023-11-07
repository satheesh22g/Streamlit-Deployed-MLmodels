import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
import seaborn as sns
from sklearn.pipeline import Pipeline
#from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
#from scipy.stats import chi2_contingency, chi2
#import statsmodels.api as sm
#from scipy.stats import spearmanr
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#from xgboost import XGBClassifier
#from scipy.stats import anderson
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from io import StringIO
matplotlib.use("Agg")


def read_csv(data):
    df = pd.read_csv(data)
    return df


def Column_information(data):

    data_info = pd.DataFrame(
                            columns=['No of observation',
                                    'No of Variables',
                                    'No of Numerical Variables',
                                    'No of Factor Variables',
                                    'No of Categorical Variables',
                                    'No of Logical Variables',
                                    'No of Date Variables',
                                    'No of zero variance variables'])

    data_info.loc[0, 'No of observation'] = data.shape[0]
    data_info.loc[0, 'No of Variables'] = data.shape[1]
    data_info.loc[0, 'No of Numerical Variables'] = data._get_numeric_data().shape[1]
    data_info.loc[0, 'No of Factor Variables'] = data.select_dtypes(
        include='category').shape[1]
    data_info.loc[0, 'No of Logical Variables'] = data.select_dtypes(
        include='bool').shape[1]
    data_info.loc[0, 'No of Categorical Variables'] = data.select_dtypes(
        include='object').shape[1]
    data_info.loc[0, 'No of Date Variables'] = data.select_dtypes(
        include='datetime64').shape[1]
    data_info.loc[0, 'No of zero variance variables'] = data.loc[:,
        data.apply(pd.Series.nunique) == 1].shape[1]

    data_info = data_info.transpose()
    data_info.columns = ['value']
    data_info['value'] = data_info['value'].astype(int)
    return data_info


def Tabulation(x):
    table = pd.DataFrame()
    table['No of Missing'] = x.isnull().sum().values
    table['No of Uniques'] = x.nunique().values
    table['Percent of Missing'] = (
        (x.isnull().sum().values) / (x.shape[0])) * 100
    return table


def outlier_count(data):
    iqr = data.quantile(q=0.85) - data.quantile(q=0.15)
    upper_out = data.quantile(q=0.85) + 1.5 * iqr
    lower_out = data.quantile(q=0.15) - 1.5 * iqr
    return len(data[data > upper_out]) + len(data[data < lower_out])


def num_summary(df):

    df_num = df._get_numeric_data()
    data_info_num = pd.DataFrame()
    i = 0
    for c in df_num.columns:
        data_info_num.loc[c,
     'Negative values count'] = df_num[df_num[c] < 0].shape[0]
        data_info_num.loc[c,
     'Positive values count'] = df_num[df_num[c] > 0].shape[0]
        data_info_num.loc[c, 'Zero count'] = df_num[df_num[c] == 0].shape[0]
        data_info_num.loc[c, 'Unique count'] = len(df_num[c].unique())
        data_info_num.loc[c, 'Negative Infinity count'] = df_num[df_num[c]
            == -np.inf].shape[0]
        data_info_num.loc[c,
     'Positive Infinity count'] = df_num[df_num[c] == np.inf].shape[0]
        data_info_num.loc[c, 'Missing Percentage'] = df_num[df_num[c].isnull(
        )].shape[0] / df_num.shape[0]
        data_info_num.loc[c, 'Count of outliers'] = outlier_count(df_num[c])
        i = i + 1
    return data_info_num


st.title("Machine Learning Application for Automated EDA")
st.markdown(
    f'''This Application is created and maintained by <a href="https://github.com/satheesh22g">*satheesh22g*</a>''',
     unsafe_allow_html=True)


activities = [
    "General EDA",
    "EDA For Linear Models",
    "EDA For Models Building",
     "Build Model"]
choice = st.sidebar.selectbox("Select Activities", activities)

if choice == 'General EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv"])
		if data is not None:
			df = read_csv(data)
			st.dataframe(df.head())
			st.success("Data Frame Loaded successfully")

			if st.checkbox("Show dtypes"):
				st.text(df.dtypes)

			if st.checkbox("Show Columns"):
				st.write(df.columns)
			if st.checkbox("Numerical Variables"):
				n_cols = [x for x in df.select_dtypes(exclude=['object'])]
				st.dataframe(df[n_cols])

			if st.checkbox("Categorical Variables"):
				c_cols = [x for x in df.select_dtypes(include=['object'])]
				st.dataframe(df[c_cols])
			if st.checkbox("Show Missing"):
				st.write(df.isna().sum())

			if st.checkbox("column information"):
				st.write(Column_information(df))

			if st.checkbox("Aggregation Tabulation"):
				st.write(Tabulation(df))

			if st.checkbox("Numerical Columns Summary"):
				st.write(num_summary(df))

			if st.checkbox("Statistical Summary"):
				st.write(df.describe().T)




			if st.checkbox("DropNA"):
				temp_df = df.dropna()
				st.dataframe(temp_df)


			if st.checkbox("Missing after DropNA"):
				st.write(temp_df.isna().sum())


			# col1 = st.selectbox("Select Column 1 For Pearsonr Correlation (Numerical Columns)",df.columns)
			# col2 = st.selectbox("Select Column 2 For Pearsonr Correlation (Numerical Columns)",df.columns)
			# if st.button("Generate Pearsonr Correlation"):
			# 	df=pd.DataFrame(pearsonr(df[col1],df[col2]))
			# 	st.dataframe(df)
			# if st.button("Generate Pearsonr Correlation"):
			# 	df=pd.DataFrame(dataframe.Show_pearsonr(imp_df[selected_columns_name3],imp_df[selected_columns_names4]),index=['Pvalue', '0'])
			# 	st.dataframe(df)

			# spearmanr3 = dataframe.show_columns(df)
			# spearmanr4 = dataframe.show_columns(df)
			# spearmanr13 = st.selectbox("Select Column 1 For spearmanr Correlation (Categorical Columns)",spearmanr4)
			# spearmanr14 = st.selectbox("Select Column 2 For spearmanr Correlation (Categorical Columns)",spearmanr4)
			# if st.button("Generate spearmanr Correlation"):
			# 	df=pd.DataFrame(dataframe.Show_spearmanr(catego_df[spearmanr13],catego_df[spearmanr14]),index=['Pvalue', '0'])
			# 	st.dataframe(df)

			st.subheader("UNIVARIATE ANALYSIS")

			all_columns_names = df.columns
			selected_columns_names = st.selectbox("Select Column for Histogram ",all_columns_names)
			if st.checkbox("Show Histogram for Selected variable"):
				fig, ax = plt.subplots()
				ax.hist(df[selected_columns_names])
				st.pyplot(fig)
			# all_columns_names = df.columns
			# selected_columns_names = st.selectbox("Select Column for Histogram ",all_columns_names,key='displot')
			# if st.checkbox("Show DisPlot for Selected variable"):

			# 	st.pyplot(sns.displot(df[selected_columns_names]))
			# all_columns_names = df.columns
			# selected_columns_names = st.selectbox("Select Column for Histogram ",all_columns_names)
			# if st.checkbox("Show Histogram for Selected variable"):
			# 	fig, ax = plt.subplots()
			# 	ax.hist(df[selected_columns_names])
			# 	st.pyplot(fig)
			# all_columns_names = df.columns
			# selected_columns_names = st.selectbox("Select Column for Histogram ",all_columns_names)
			# if st.checkbox("Show Histogram for Selected variable"):
			# 	fig, ax = plt.subplots()
			# 	ax.hist(df[selected_columns_names])
			# 	st.pyplot(fig)
			# all_columns_names = dataframe.show_columns(df)
			# selected_columns_names = st.selectbox("Select Columns Distplot ",all_columns_names)
			# if st.checkbox("Show DisPlot for Selected variable"):
			# 	st.write(plt.Show_DisPlot(df[selected_columns_names]))
			# 	st.pyplot()

			# all_columns_names = dataframe.show_columns(df)
			# selected_columns_names = st.selectbox("Select Columns CountPlot ",all_columns_names)
			# if st.checkbox("Show CountPlot for Selected variable"):
			# 	st.write(dataframe.Show_CountPlot(df[selected_columns_names]))
			# 	st.pyplot()

			# st.subheader("BIVARIATE ANALYSIS")

			# Scatter1 = dataframe.show_columns(df)
			# Scatter2 = dataframe.show_columns(df)
			# Scatter11 = st.selectbox("Select Column 1 For Scatter Plot (Numerical Columns)",Scatter1)
			# Scatter22 = st.selectbox("Select Column 2 For Scatter Plot (Numerical Columns)",Scatter2)
			# if st.button("Generate PLOTLY Scatter PLOT"):
			# 	st.pyplot(dataframe.plotly(df,df[Scatter11],df[Scatter22]))

			# bar1 = dataframe.show_columns(df)
			# bar2 = dataframe.show_columns(df)
			# bar11 = st.selectbox("Select Column 1 For Bar Plot ",bar1)
			# bar22 = st.selectbox("Select Column 2 For Bar Plot ",bar2)
			# if st.button("Generate PLOTLY histogram PLOT"):
			# 	st.pyplot(dataframe.plotly_histogram(df,df[bar11],df[bar22]))

			# violin1 = dataframe.show_columns(df)
			# violin2 = dataframe.show_columns(df)
			# violin11 = st.selectbox("Select Column 1 For violin Plot",violin1)
			# violin22 = st.selectbox("Select Column 2 For violin Plot",violin2)
			# if st.button("Generate PLOTLY violin PLOT"):
			# 	st.pyplot(dataframe.plotly_violin(df,df[violin11],df[violin22]))

			# st.subheader("MULTIVARIATE ANALYSIS")

			# if st.checkbox("Show Histogram"):
			# 	st.write(dataframe.show_hist(df))
			# 	st.pyplot()

			# if st.checkbox("Show HeatMap"):
			# 	st.write(dataframe.Show_HeatMap(df))
			# 	st.pyplot()

			# if st.checkbox("Show PairPlot"):
			# 	st.write(dataframe.Show_PairPlot(df))
			# 	st.pyplot()

			# if st.button("Generate Word Cloud"):
			# 	st.write(dataframe.wordcloud(df))
			# 	st.pyplot()