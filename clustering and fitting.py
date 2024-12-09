#clustering and fitting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset (adjust path as needed)
data = pd.read_csv("drug200.csv")

# Preprocess the data
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
categorical_columns = ['Sex', 'BP', 'Cholesterol', 'Drug']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Plot 1: Histogram of Age
plt.figure(figsize=(8, 5))
plt.hist(data['Age'], bins=10, color='orange', edgecolor='blue')
plt.title("Histogram of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Plot 2: Scatter plot of Age vs Na_to_K
plt.figure(figsize=(8, 5))
plt.scatter(data['Age'], data['Na_to_K'], color='cyan', alpha=0.7)
plt.title("Scatter Plot of Age vs Na_to_K")
plt.xlabel("Age")
plt.ylabel("Na_to_K")
plt.grid()
plt.show()

# Plot 3: Heatmap of Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Correlation Matrix")
plt.show()

# Plot 4: Elbow Plot for K-Means Clustering
X = data[['Age', 'Na_to_K']]
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='-', color='green')
plt.title("Elbow Plot for Optimal Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid()
plt.show()
