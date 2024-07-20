import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import seaborn as sns
from datetime import datetime

# Read the CSV file 
df = pd.read_csv("SocialMediaUsersDataset.csv")
df.head()

df.shape

# Convert interests and name to suitable format for the apriori algorithm
df['Name'] = df['Name'].apply(lambda x: [i.strip() for i in x.strip('[]"').split(',')])
df['Interests'] = df['Interests'].apply(lambda x: [i.strip() for i in x.strip('[]"').split(',')])

df['Interests_Name'] = df['Interests'] + df['Name']

# Convert DOB to datetime and calculate age
df['DOB'] = pd.to_datetime(df['DOB'], format='%Y-%m-%d')
current_date = datetime.now()

# I created the Age from the Current Date - Date of Birth to have a integer number
df['Age'] = df['DOB'].apply(lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day)))

# Encode Gender and Country
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Country'] = label_encoder.fit_transform(df['Country'])

# Check the data types of UserID and Age 
print("Data types of the columns:")
print(df[['UserID', 'Age','Gender','Country']].dtypes)

data = list(zip('Gender','Country'))

# Apply TransactionEncoder to encode the interests
te = TransactionEncoder()
encoded_interests = te.fit_transform(df['Interests_Name'])
df_encoded = pd.DataFrame(encoded_interests, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.10, use_colnames=True)

#Let's view our interpretation values using the Associan rule function.
df_ar = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
df_ar

print('Apriori algorithm result:')
print(frequent_itemsets)

# Defien the inputs we will use for our K-mean clustering algorithm
X = df[['UserID','Age']].copy()

# This loop determine the number of Python clusters that we will use
cluster_variances = []
K = range(1, 15)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    cluster_variances.append(kmeans.inertia_)

# Plot the elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(K, cluster_variances,color='pink', marker='x',linestyle='-')
plt.xlabel('Number of clusters')
plt.ylabel('Cluster Variances')
plt.title('Elbow Method')
plt.show()

young_age = df[(df['Age'] >= 18) & (df['Age'] <= 35)]
middle_age = df[(df['Age'] > 35) & (df['Age'] <= 55)]
old_age = df[df['Age'] > 55]

# Plot the histogram with different colors for each category
sns.histplot(data=young_age, x='Age', bins=2, color='blue', label='Young')
sns.histplot(data=middle_age, x='Age', bins=2, color='orange', label='Middle-aged')
sns.histplot(data=old_age, x='Age', bins=2, color='green', label='Old')

# Add title and labels
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Show legend
plt.legend()

# Show the plot
plt.show()




