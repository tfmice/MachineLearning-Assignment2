import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

dataset = pd.read_csv("listings.csv", delimiter=",")
dataset["reviews_per_month"] = dataset["reviews_per_month"].replace(",", ".", regex=True)
dataset = dataset.dropna()
dataset = dataset.astype({"price" : "int32", "minimum_nights" : "int32", "number_of_reviews" : "int32", "calculated_host_listings_count" : "int32", "availability_365" : "int32", "reviews_per_month" : "float64"})
dataset["last_review"] = pd.to_datetime(dataset["last_review"], format= "%Y-%m-%d")
print(dataset.dtypes)
print(dataset.isna().values.any())
print(dataset.head())


newDataset = dataset[["neighbourhood", "room_type", "price", "minimum_nights", "number_of_reviews", "last_review", "reviews_per_month", "calculated_host_listings_count", "availability_365"]]
newDataset.hist()
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.rcParams["figure.figsize"] = [16,9]
plt.show()


plt.subplot(331)
plt.scatter(newDataset["room_type"], newDataset["price"], color="blue")
plt.xlabel("room_type")
plt.ylabel("price")
plt.subplot(332)
plt.scatter(newDataset["minimum_nights"], newDataset["price"], color="blue")
plt.xlabel("minimum_nights")
plt.ylabel("price")
plt.subplot(333)
plt.scatter(newDataset["neighbourhood"], newDataset["price"], color="blue")
plt.xlabel("neighbourhood")
plt.ylabel("price")
plt.subplot(334)
plt.scatter(newDataset["number_of_reviews"], newDataset["price"], color="blue")
plt.xlabel("number_of_reviews")
plt.ylabel("price")
plt.subplot(335)
plt.scatter(newDataset["last_review"], newDataset["price"], color="blue")
plt.xlabel("last_review")
plt.ylabel("price")
plt.subplot(336)
plt.scatter(newDataset["reviews_per_month"], newDataset["price"], color="blue")
plt.xlabel("reviews_per_month")
plt.ylabel("price")
plt.subplot(337)
plt.scatter(newDataset["calculated_host_listings_count"], newDataset["price"], color="blue")
plt.xlabel("calculated_host_listings_count")
plt.ylabel("price")
plt.subplot(338)
plt.scatter(newDataset["availability_365"], newDataset["price"], color="blue")
plt.xlabel("availability_365")
plt.ylabel("price")
plt.subplots_adjust(hspace = 0.5, wspace = 0.1)
plt.rcParams["figure.figsize"] = [22,9]
plt.show()


train, test = train_test_split(newDataset, test_size=0.2)
regression = linear_model.LinearRegression()
regression.fit(train[["number_of_reviews"]], train[["price"]])
print('Coefficients: ', regression.coef_)
print('Intercept: ',regression.intercept_)
print('Train: ',len(train))
print('Test: ',len(test))


plt.scatter(train["number_of_reviews"], train["price"],  color='blue')
plt.plot(train[["number_of_reviews"]], regression.coef_ * train[["number_of_reviews"]] + regression.intercept_, '-r')
plt.xlabel("number_of_reviews")
plt.ylabel("price")
plt.rcParams["figure.figsize"] = [9,7]
plt.show()


sb.pairplot(train)
sb.lmplot("number_of_reviews", "price", data = train)
plt.show()


prediction = regression.predict(test[["number_of_reviews"]])
for i in range(len(test)):
    print(test[["number_of_reviews"]].values[i], prediction[i])
print("MAE : ", mean_absolute_error(test[["price"]], prediction))
print("MSE : ", mean_squared_error(test[["price"]], prediction))
print("R2 : ", r2_score(test[["price"]], prediction))