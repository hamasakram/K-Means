import streamlit as st
import io  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read data
dt = pd.read_csv("Mall_Customers.csv")
st.write("Data Read!")

# Display the head of the DataFrame
st.write(dt.head())

# Display information about the DataFrame
buffer = io.StringIO()
dt.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Check for missing values in the dataset
st.write(dt.isnull().sum())

# Display histograms for age, income and spending score
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
dt[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].hist(bins=15, ax=ax)
st.pyplot(fig)

# Display count plot for gender
fig, ax = plt.subplots()
sns.countplot(x='Gender', data=dt)
plt.title('Distribution of Gender')
st.pyplot(fig)

# Mapping gender to numerical values
dt['Gender'] = dt['Gender'].map({'Female': 0, 'Male': 1})

# Pairplot to visualize the relationships between Age, Annual Income, and Spending Score
st.write("Pairplot and correlation matrix will take a moment to load...")
pairplot_fig = sns.pairplot(dt, hue='Gender', vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
st.pyplot(pairplot_fig)

# Correlation matrix
corr_fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(dt.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Between Features')
st.pyplot(corr_fig)

# Boxplots to check for outliers in the numerical columns
for column in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(dt[column], ax=ax)
    ax.set_title(f'Boxplot of {column}')
    st.pyplot(fig)

# Data scaling and clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(dt[['Annual Income (k$)', 'Spending Score (1-100)']])
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
dt['Cluster'] = clusters

# Plotting clusters
cluster_fig, ax = plt.subplots(figsize=(10, 6))
colors = ['yellow', 'purple', 'blue']
descriptions = ['Low-income, Low-spending', 'Moderate Income, Low Spending', 'High Income, High Spending']
for cluster_label, color, description in zip([0, 1, 2], colors, descriptions):
    cluster_data = dt[dt['Cluster'] == cluster_label]
    ax.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
               c=color, label=f'{description} (Cluster {cluster_label})')
ax.set_title('Customer Segmentation by Income and Spending')
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.legend()
st.pyplot(cluster_fig)

# Silhouette score
from sklearn.metrics import silhouette_score
score = silhouette_score(data_scaled, clusters)
st.write('Silhouette Score: {:.2f}'.format(score))

# Function for predicting customer cluster
def predict_customer_cluster(features, scaler, model):
    scaled_features = scaler.transform([features])
    cluster = model.predict(scaled_features)
    centroid = model.cluster_centers_[cluster[0]]
    dist = np.linalg.norm(scaled_features - centroid)
    max_dist = 3
    confidence = max(0, 1 - dist / max_dist)
    return cluster[0], confidence

# User input for prediction
income = st.number_input("Enter annual income (k$):", min_value=0)
spending_score = st.number_input("Enter spending score (1-100):", min_value=0)
if st.button("Predict Cluster"):
    cluster_label, confidence = predict_customer_cluster([income, spending_score], scaler, kmeans)
    st.write(f'The customer belongs to cluster: {cluster_label} with confidence: {confidence:.2f}')
