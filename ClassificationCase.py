import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("listings.csv", delimiter=",")
dataset["reviews_per_month"] = dataset["reviews_per_month"].replace(",", ".", regex=True)
dataset = dataset.dropna()
dataset = dataset.astype({"price" : "int32", "minimum_nights" : "int32", "number_of_reviews" : "int32", "calculated_host_listings_count" : "int32", "availability_365" : "int32", "reviews_per_month" : "float64"})
dataset["last_review"] = pd.to_datetime(dataset["last_review"], format= "%Y-%m-%d")
print(dataset["room_type"].unique())
dataset["room_type"] = pd.Categorical(dataset["room_type"], dataset["room_type"].unique())
dataset["room_type"] = dataset["room_type"].cat.rename_categories([1,2,3])
print(dataset.dtypes)
print(dataset.isna().values.any())
print(dataset.head())

train, test = train_test_split(dataset, test_size=0.2)

Ks = 10
accuracy = np.zeros((Ks-1))
ConfustionMx = []
for n in range(1, Ks):    
    KNN = KNeighborsClassifier(n_neighbors = n).fit(train[["price", "number_of_reviews", "minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365"]], train["room_type"])  
    classification = KNN.predict(test[["price", "number_of_reviews", "minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365"]])
    accuracy[n - 1] = accuracy_score(test["room_type"], classification)

print("Best  ACC : %.2f" % accuracy.max(), ", with k = ", accuracy.argmax() + 1)