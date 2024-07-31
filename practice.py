#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif


# In[8]:


# Load the datasets
google_ads_df = pd.read_csv('googlead-performance.csv')
meta_ads_df = pd.read_csv('metaads-performance.csv')
microsoft_ads_df = pd.read_csv('microsoftads-performance.csv')
website_landings_df = pd.read_csv('website-landings.csv')


# In[9]:


print(google_ads_df.head())


# In[10]:


print(meta_ads_df.head())


# In[11]:


print(microsoft_ads_df.head())


# In[12]:


print(website_landings_df.head())


# In[13]:


# Convert date columns to datetime
google_ads_df['Date'] = pd.to_datetime(google_ads_df['Date'])
meta_ads_df['Date'] = pd.to_datetime(meta_ads_df['Date'])
microsoft_ads_df['Date'] = pd.to_datetime(microsoft_ads_df['Date'])
website_landings_df['Website Landing Time'] = pd.to_datetime(website_landings_df['Website Landing Time'])


# In[14]:


google_ads_df.fillna(0, inplace=True)
meta_ads_df.fillna(0, inplace=True)
microsoft_ads_df.fillna(0, inplace=True)
website_landings_df.fillna(0, inplace=True)


# In[15]:


google_ads_df['CTR'] = google_ads_df['Clicks'] / google_ads_df['Impressions']
google_ads_df['CPC'] = google_ads_df['Cost'] / google_ads_df['Clicks'].replace(0, np.nan)  # Handle division by zero

meta_ads_df['CTR'] = meta_ads_df['Clicks'] / meta_ads_df['Impressions']
meta_ads_df['CPC'] = meta_ads_df['Cost'] / meta_ads_df['Clicks'].replace(0, np.nan)  # Handle division by zero

microsoft_ads_df['CTR'] = microsoft_ads_df['Clicks'] / microsoft_ads_df['Impressions']
microsoft_ads_df['CPC'] = microsoft_ads_df['Cost'] / microsoft_ads_df['Clicks'].replace(0, np.nan)  # Handle division by zero


# In[16]:


# Group by the correct column names and aggregate the data
try:
    landing_agg = website_landings_df.groupby(['Source', 'Channel', 'Campaign Type']).agg({
        'User Id': 'count',
        'Is Converted': 'sum'
    }).reset_index().rename(columns={'User Id': 'Total Landings', 'Is Converted': 'Total Conversions'})

    print("\nAggregated Website Landings Data:")
    print(landing_agg.head())
except KeyError as e:
    print(f"KeyError: {e}. Please check the column names and update the code accordingly.")


# In[17]:


print(website_landings_df.describe(include='all'))  # Summary statistics for all columns


# In[18]:


# Convert 'Campaign Type' to string if it's categorical
website_landings_df['Campaign Type'] = website_landings_df['Campaign Type'].astype(str)

# Clean 'Source' column by removing or correcting placeholder values
website_landings_df['Source'] = website_landings_df['Source'].replace('0.0', np.nan)

# Drop rows where 'Source' or 'Campaign Type' is NaN if necessary
website_landings_df.dropna(subset=['Source', 'Campaign Type'], inplace=True)

# Recheck data types
print(website_landings_df.dtypes)

# Aggregate website landings data
landing_agg = website_landings_df.groupby(['Source', 'Channel', 'Campaign Type']).agg({
    'User Id': 'count',
    'Is Converted': 'sum'
}).reset_index().rename(columns={'User Id': 'Total Landings', 'Is Converted': 'Total Conversions'})

print("\nAggregated Website Landings Data:")
print(landing_agg.head())


# In[19]:


print(website_landings_df.describe(include='all'))  # Summary statistics for all columns


# In[20]:


google_ads_df['CTR'] = google_ads_df['Clicks'] / google_ads_df['Impressions'] #Click through Rate
google_ads_df['CPC'] = google_ads_df['Cost'] / google_ads_df['Clicks']    #Cost Per Click


# In[21]:


meta_ads_df['CTR'] = meta_ads_df['Clicks'] / meta_ads_df['Impressions']
meta_ads_df['CPC'] = meta_ads_df['Cost'] / meta_ads_df['Clicks']

microsoft_ads_df['CTR'] = microsoft_ads_df['Clicks'] / microsoft_ads_df['Impressions']
microsoft_ads_df['CPC'] = microsoft_ads_df['Cost'] / microsoft_ads_df['Clicks']


# In[22]:


landing_agg = website_landings_df.groupby(['Source', 'Channel', 'Campaign Type']).agg({
    'User Id': 'count',
    'Is Converted': 'sum'
}).reset_index().rename(columns={'User Id': 'Total Landings', 'Is Converted': 'Total Conversions'})


# In[23]:


print("\nAggregated Website Landings Data:")
print(landing_agg.head())


# In[24]:


features = ['Impressions', 'Clicks', 'Cost', 'CTR', 'CPC']
target = 'Conversions'

X = google_ads_df[features]
y = google_ads_df[target]


# In[25]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[28]:


from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor

# Train your model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)



# In[30]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Example predictions and actual values (replace with your data)
# y_test = actual values
# predictions = model predictions

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Random Forest Model RMSE: {rmse:.2f}")

# Calculate MAE
mae = mean_absolute_error(y_test, predictions)
print(f"Random Forest Model MAE: {mae:.2f}")


# In[31]:


feature_importances = model.feature_importances_
for feature, importance in zip(features, feature_importances):
    print(f"Feature: {feature}, Importance: {importance:.4f}")



# In[32]:


# Customer Segmentation using KMeans
# kmeans = KMeans(n_clusters=5, random_state=42)
# website_landings_df['Cluster'] = kmeans.fit_predict(website_landings_df[['Is Converted']])
# Example of additional features
features = ['Is Converted', 'Source', 'Channel']  # Add more relevant features
X = pd.get_dummies(website_landings_df[features])  # Convert categorical features to numerical

kmeans = KMeans(n_clusters=5, random_state=42)
website_landings_df['Cluster'] = kmeans.fit_predict(X)

print("\nCustomer Segmentation:")
print(website_landings_df.head())



# In[33]:


# Check the number of data points in each cluster
cluster_counts = website_landings_df['Cluster'].value_counts()
print("\nCluster Sizes:")
print(cluster_counts)


# In[34]:


# Sample a smaller subset of the data
#small data for debugging
sample_df = website_landings_df.sample(frac=0.1, random_state=42)  # Adjust fraction as needed


# In[37]:


# In[ ]:


# Testing different number of clusters
for n_clusters in range(2, 11):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    sample_df['Cluster'] = kmeans.fit_predict(sample_df[['Is Converted']])
    silhouette_avg = silhouette_score(sample_df[['Is Converted']], sample_df['Cluster'])
    print(f'Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg}')


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Include additional features and convert categorical features to numerical
features = ['Is Converted', 'Source', 'Channel']
X = pd.get_dummies(website_landings_df[features], drop_first=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=website_landings_df['Cluster'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Cluster')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Clusters')
plt.show()


# In[ ]:


from sklearn.cluster import DBSCAN

# Apply DBSCAN for clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
website_landings_df['Cluster'] = dbscan.fit_predict(website_landings_df[['Is Converted']])

# Check cluster sizes
cluster_counts_dbscan = website_landings_df['Cluster'].value_counts()
print("\nDBSCAN Cluster Sizes:")
print(cluster_counts_dbscan)


# In[ ]:


website_landings_df['Is Converted'].value_counts().plot(kind='bar')
plt.title('Distribution of Conversion Status')
plt.xlabel('Is Converted')
plt.ylabel('Count')
plt.show()


# In[ ]:


cluster_counts = website_landings_df['Cluster'].value_counts()
print("\nCluster Sizes:")
print(cluster_counts)


# In[ ]:


print(google_ads_df.head())


# In[ ]:




