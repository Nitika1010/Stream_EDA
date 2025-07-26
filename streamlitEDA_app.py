import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Customer Churn Analysis Dashboard")

# Load data
df = pd.read_csv("Customer Churn.csv")

# Data Cleaning
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
df["TotalCharges"] = df["TotalCharges"].astype("float")
df["SeniorCitizen"] = df["SeniorCitizen"].apply(lambda x: "Yes" if x == 1 else "No")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head(30))

# Data Info
st.subheader("Data Overview")
st.write("Shape of the dataset:", df.shape)
st.write("Missing values:", df.isnull().sum())
st.write("Duplicate Customer IDs:", df["customerID"].duplicated().sum())
st.write(df.describe())

# Countplot of Churn
st.subheader("Churn Count")
fig1, ax1 = plt.subplots(figsize=(5, 5))
ax = sns.countplot(x="Churn", data=df)
ax.bar_label(ax.containers[0])
plt.title("Count of Customers BY Churn")
st.pyplot(fig1)

# Pie Chart for Churn
st.subheader("Churn Percentage")
fig2, ax2 = plt.subplots(figsize=(4, 4))
gb = df.groupby("Churn").agg({'Churn': "count"})
ax2.pie(gb["Churn"], labels=gb.index, autopct="%1.2f%%")
plt.title("Percentage of Churn Customers")
st.pyplot(fig2)

# Gender-wise churn
st.subheader("Churn by Gender")
fig3, ax3 = plt.subplots(figsize=(4, 4))
sns.countplot(x="gender", data=df, hue="Churn", ax=ax3)
plt.title("Churn By Gender")
st.pyplot(fig3)

# Senior Citizen Count
st.subheader("Senior Citizen Distribution")
fig4, ax4 = plt.subplots(figsize=(4, 5))
ax = sns.countplot(x="SeniorCitizen", data=df)
ax.bar_label(ax.containers[0])
plt.title("Count of Customers by Senior Citizen")
st.pyplot(fig4)

# Senior Citizen Churn %
st.subheader("Churn by Senior Citizen")
total_count = df.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).unstack()
fig5, ax5 = plt.subplots(figsize=(5, 5))
total_count.plot(kind='bar', stacked=True, ax=ax5, color=['#1f77b4', '#ff7f0e'])

for p in ax5.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax5.text(x + width / 2, y + height / 2, f'{height * 100:.1f}%', ha='center', va='center')

plt.title("Churn By Senior Citizen")
plt.xlabel("SeniorCitizen")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title='Churn', loc='upper right')
plt.tight_layout()
st.pyplot(fig5)

# Tenure Histogram
st.subheader("Tenure Distribution by Churn")
fig6, ax6 = plt.subplots(figsize=(10, 5))
sns.histplot(x="tenure", data=df, bins=75, hue="Churn", ax=ax6)
st.pyplot(fig6)

# Contract vs Churn
st.subheader("Churn by Contract Type")
fig7, ax7 = plt.subplots(figsize=(5, 5))
ax = sns.countplot(x="Contract", data=df, hue="Churn")
ax.bar_label(ax.containers[0])
plt.title("Count of Customers By Contract")
st.pyplot(fig7)

# Service-based Churn
st.subheader("Churn by Services Subscribed")
columns = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

n_cols = 3
n_rows = (len(columns) + n_cols - 1) // n_cols
fig8, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.countplot(data=df, x=col, hue='Churn', ax=axes[i])
    axes[i].set_title(f'{col} by Churn')
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes)):
    fig8.delaxes(axes[j])

plt.tight_layout()
st.pyplot(fig8)

# Churn by Payment Method
st.subheader("Churn by Payment Method")
fig9, ax9 = plt.subplots(figsize=(10, 5))
ax = sns.countplot(x="PaymentMethod", data=df, hue="Churn")
ax.bar_label(ax.containers[0])
plt.title("Churned Customers By Payment Method")
st.pyplot(fig9)

# Summary
st.markdown("""---""")
st.subheader("Key Insights")
st.markdown("""
- About **26.54%** of customers have churned.
- **Month-to-month contract** users are more likely to churn than those with long-term contracts.
- Customers **not using OnlineSecurity, TechSupport, and DeviceProtection** are more likely to churn.
- **Senior Citizens** show higher churn percentage than non-seniors.
- **Electronic check** users show higher churn rates.
""")
