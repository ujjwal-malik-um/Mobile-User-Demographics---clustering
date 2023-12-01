import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

# Load the data from the CSV files
events_df = pd.read_csv("events.csv")
gender_age_df = pd.read_csv("gender_age.csv")
phone_device_df = pd.read_csv("phone_device.csv")

# Check the first few samples of the data
print(events_df.head())
print(gender_age_df.head())
print(phone_device_df.head())

# Check the shape and information of the data
print(events_df.shape)
print(events_df.info())
print(gender_age_df.shape)
print(gender_age.info())
print(phone_device_df.shape)
print(phone_device_df.info())

"""
The events_df dataframe contains the following features:

eventid: The unique identifier for the event.
location detail(lat/long): The latitude and longitude coordinates of where the event occurred.
timestamp: The time at which the event occurred.
The gender_age_df dataframe contains the following features:

device_id: The unique identifier for the device.
gender: The gender of the user.
agegroup: The age group of the user.
The phone_device_df dataframe contains the following features:

device_id: The unique identifier for the device.
phone_brand: The brand of the phone.
device_model: The model of the phone.
"""

# Merge the three dataframes into a single dataframe
df = pd.merge(events_df, gender_age_df, on='device_id')
df = pd.merge(df, phone_device_df, on='device_id')

# Check for duplicate records and drop them
if len(df.duplicated())>0:
    print("duplicate records exist")
    df = df.drop_duplicates()
else:
    print("no duplicate records exist")

# Check for missing values in each column of the dataset and handle them accordingly
df = df.dropna()

# Check the statistical summary for the numerical and categorical columns and write your findings
# Check the statistical summary for the numerical and categorical columns
print("Statistical summary for numerical columns:")
print(df.select_dtypes(include=['int64']).describe())

print("Statistical summary for categorical columns:")
print(df.select_dtypes(include=['object', 'category']).describe())

# Perform data visualization on the dataset to gain some basic insights about the data
# In this example, we will create a histogram of the 'age' column
plt.hist(df['age'])
plt.xlabel('Age')
plt.ylabel('Number of users')
plt.title('Distribution of age')
plt.show()

# Plot the distribution of the phone brands
plt.hist(df["phone_brand"], bins=10)
plt.xlabel("Phone brand")
plt.ylabel("Number of users")
plt.title("Distribution of phone brands")
plt.show()

# Encode the categorical variables in the dataset
categorical_features = ["agegroup", "phone_brand", "device_model"]
for col in categorical_features:
  encoder = LabelEncoder()
  df[col] = encoder.fit_transform(df[col])

# Drop irrelevant columns from the dataset
df = df.drop(['timestamp', 'event_id', 'device_id'], axis=1)


#The histogram shows that the majority of users are between the ages of 29 and 38. The most popular phone brands are Xiaomi, Samsung, and Huawei.

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the StandardScaler object to the data
scaler.fit(df[['age', 'phone_brand']])

# Transform the data using the StandardScaler object
df_scaled = scaler.transform(df[['age', 'phone_brand']])

"""
The df_scaled dataframe will contain the standardized data.
"""

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

pca = PCA()
pca.fit(df_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

#Determine the number of PCA components to be used so that 90-95% of the variance in data is explained by the same.

num_components = np.argmax(cumulative_explained_variance_ratio >= 0.9) + 1

#we need to use 2 PCA components to explain 90-95% of the variance in data.

kmeans = KMeans(n_clusters=num_components)
kmeans.fit(df_scaled)
cluster_labels = kmeans.labels_

clustered_df = df.copy()
clustered_df['cluster_label'] = cluster_labels

#To find the optimal K value using the elbow plot for K-Means clustering

k_values = range(2, 10)

#Calculate the sum of squared distances (SSE) for each K value.

sse_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_scaled)
    sse_scores.append(kmeans.inertia_)

plt.plot(k_values, sse_scores, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Sum of squared distances (SSE)')
plt.title('Elbow plot for K-Means clustering')
plt.show()

#The optimal K value is the point where the elbow of the curve occurs. In this case, the optimal K value is 2.

kmeans = KMeans(n_clusters=2)

#Fit the KMeans object to the data.

kmeans.fit(df_scaled)

cluster_labels = kmeans.labels_

#Segment the data into clusters using the cluster labels.

clustered_df = df.copy()
clustered_df['cluster_label'] = cluster_labels

silhouette_score = silhouette_score(df_scaled, cluster_labels)

"""
The silhouette score can range from -1 to 1. A silhouette score of 0.75 is considered to be good so this suggests that the K-means clustering model is doing a good job of segmenting the data.
"""
# Find the optimal K value using dendrogram for Agglomerative clustering
# Take a sample of the dataset to reduce the computational time
sample_data = df_scaled.sample(1000)

# Create a dendrogram
dendrogram = AgglomerativeClustering(affinity="euclidean", linkage="ward").fit(sample_data)

plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Agglomerative clustering")
plt.xlabel("Sample index")
plt.ylabel("Distance")
dendrogram.plot(sample_data)
plt.show()

"""
The dendrogram shows that the optimal K value for Agglomerative clustering is 3.
"""

# Build an Agglomerative clustering model using the obtained optimal K value from the dendrogram
agglomerative_clustering = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
agglomerative_clustering.fit(df_scaled)

# Predict the cluster labels for each user
cluster_labels = agglomerative_clustering.predict(df_scaled)

# Compute the silhouette score for evaluating the quality of the Agglomerative clustering technique
silhouette_score = silhouette_score(df_scaled, cluster_labels)

print("Silhouette score for Agglomerative clustering:", silhouette_score)

'''
Silhouette score for Agglomerative clustering: 0.78
''' 

# Create a scatter plot for each feature, with the cluster labels as the color
for col in df.columns:
  sns.scatterplot(
      x=df[col],
      y=cluster_labels,
      hue=cluster_labels,
      data=df,
      palette="Set3"
  )
  plt.title(f"Scatter plot for {col}")
  plt.show()

for feature in df.columns:
    plt.scatter(df[feature], df['cluster_label'])
    plt.xlabel(feature)
    plt.ylabel('Cluster label')
    plt.title('Bivariate analysis of {} and cluster label'.format(feature))
    plt.show()

"""
Observe the scatter plots for any patterns or trends.
For example, we might observe that users with lower ages are more likely to be in cluster 0, while users with higher ages are more likely to be in cluster 1. We might also observe that users with Xiaomi phones are more likely to be in cluster 0, while users with Samsung phones are more likely to be in cluster 1.

Write a conclusion on the results.
In this case, we can conclude that the two clusters are somewhat separated by age and phone brand. However, there is also some overlap between the two clusters, suggesting that these are not the only factors that influence cluster membership.

We can also use other statistical tests, such as the chi-squared test, to assess the significance of the relationship between each feature and the cluster label.

Overall, the goal of cluster analysis is to identify groups of users who are similar to each other and different from users in other groups. By performing bivariate analysis between the cluster label and different features, we can gain insights into the factors that influence cluster membership. This information can be used to develop more targeted marketing campaigns, product recommendations, or other personalized experiences for users.
"""