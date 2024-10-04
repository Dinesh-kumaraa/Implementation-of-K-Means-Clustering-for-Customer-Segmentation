# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: START

STEP 2: Import necessary libraries and load the customer dataset using pandas. Preview the dataset using data.head().

STEP 3: Check for missing values and inspect the dataset structure using data.info() and data.isnull().sum().

STEP 4: Import the KMeans algorithm from sklearn.cluster and initialize an empty list wcss to store inertia values (Within-Cluster Sum of Squares).

STEP 5: Use a for loop (from 1 to 10) to apply KMeans with different numbers of clusters. Fit the model to the relevant columns and store the WCSS values for each iteration in wcss.

STEP 6: Plot the WCSS values to visualize the "Elbow Method" and determine the optimal number of clusters.

STEP 7: Apply KMeans with the chosen number of clusters (5), predict cluster labels, and add the results to a new column cluster in the dataset.

STEP 8: Visualize the clusters using a scatter plot, separating data points by their cluster labels.

STEP 9: END


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: DINESH KUMARAA K
RegisterNumber: 212222220012
*/

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/LENOVO/Downloads/Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = [] #Within-Cluster Sum of Square
#It is the sum of squared distance between each point & the centroid in a cluster.

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
Elbow Method
![image](https://github.com/user-attachments/assets/13fd413c-ecfe-4dac-bc8e-7cb5457cefcf)

KMeans
![image](https://github.com/user-attachments/assets/2aa8cfd4-bdec-4014-b8d1-a58620843d9f)

y_prediction
![image](https://github.com/user-attachments/assets/bac04dce-990a-4359-ad03-ff88cefd1f29)

Customer_Segments
![image](https://github.com/user-attachments/assets/6b524e8a-59c1-47fa-b7bb-2122feafe0da)






## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
