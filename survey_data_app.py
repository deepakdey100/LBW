
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Survey Data Analysis", layout="wide")

st.title("📊 Survey Data Analysis App")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Merged_Harmonized_Survey_Data.csv")

df = load_data()

st.subheader("1. Data Preview")
st.dataframe(df.head())

# Correlation heatmap
st.subheader("2. Correlation Heatmap of Scaled Indicators")
scaled_cols = [col for col in df.columns if col.endswith('_scaled')]
fig, ax = plt.subplots()
sns.heatmap(df[scaled_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Histogram
st.subheader("3. Distribution of Scaled Low Birth Weight")
fig, ax = plt.subplots()
sns.histplot(df['LBW_scaled'], bins=5, kde=True, ax=ax)
ax.set_title("Distribution of LBW (scaled)")
st.pyplot(fig)

# Linear Regression
st.subheader("4. Linear Regression: Predict LBW from Other Indicators")
features = ['CSECTION_scaled', 'CHILD_POP_scaled', 'TFR_scaled', 'GDP_PC_scaled', 'GNI_PC_scaled']
X = df[features]
y = df['LBW_scaled']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.markdown(f"**MSE:** {mse:.4f}")
st.markdown(f"**R² Score:** {r2:.4f}")

fig, ax = plt.subplots()
sns.scatterplot(x=y, y=y_pred, ax=ax)
ax.set_xlabel("Actual LBW_scaled")
ax.set_ylabel("Predicted LBW_scaled")
ax.set_title("Actual vs Predicted LBW")
st.pyplot(fig)

# Clustering
st.subheader("5. KMeans Clustering")
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

fig, ax = plt.subplots()
sns.scatterplot(data=df, x='CSECTION_scaled', y='LBW_scaled', hue='Cluster', palette='Set2', ax=ax)
ax.set_title("Clusters by C-section and LBW Rates")
st.pyplot(fig)

# PCA
st.subheader("6. PCA Visualization")
pca = PCA(n_components=2)
df[['PC1', 'PC2']] = pca.fit_transform(X)

fig, ax = plt.subplots()
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
ax.set_title("PCA of Countries Based on Scaled Indicators")
st.pyplot(fig)

st.subheader("7. Final Data Table")
st.dataframe(df[['Country_name', 'LBW_scaled', 'Cluster', 'PC1', 'PC2']])
