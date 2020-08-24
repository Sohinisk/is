import pandas as pd
import seaborn as sn
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt


data_set = pd.read_csv(r"data.csv",engine='python',encoding = "ISO-8859-1", error_bad_lines=False)
data_set = data_set.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "TransmissionType": "Transmission",
                                    "Driven_Wheels": "Drive Mode","highway MPG": "MPG_H", "city mpg": "MPG_C",
                                    "MSRP": "Price"})
data_set = data_set.dropna()

plt.hist(data_set.Price)
plt.show()
lower_bound = 0.1
upper_bound=0.99
min_th,max_th = data_set.Price.quantile([lower_bound,upper_bound])
print(min_th,max_th)
data_set= data_set[(data_set.Price>min_th) & (data_set.Price<max_th)]

plt.hist(data_set.MPG_H)
plt.show()
min_th,max_th = data_set.MPG_H.quantile([lower_bound,upper_bound])
data_set= data_set[(data_set.MPG_H>min_th) & (data_set.MPG_H<max_th)]
plt.hist(data_set.MPG_H)
plt.show()


plt.hist(data_set.MPG_C)
plt.show()
min_th,max_th = data_set.MPG_C.quantile([lower_bound,upper_bound])
data_set= data_set[(data_set.MPG_C>min_th) & (data_set.MPG_C<max_th)]

brands=data_set['Make'].value_counts()[:5].index.tolist()
data_set2=data_set[data_set['Make'].isin(brands)]
mean = data_set2['Price'].mean()

corelation_matrix = data_set.corr()
print(corelation_matrix)

sn.heatmap(corelation_matrix, annot=True)
sn.pairplot(corelation_matrix)
plt.show()

sn.relplot(y='Number of Doors',x='Price',data=data_set)
plt.show()
sn.relplot(y='Year',x='Price',data=data_set)
plt.show()
sn.relplot(y='HP',x='Price',data=data_set)
plt.show()
sn.relplot(y='Cylinders',x='Price',data=data_set)
plt.show()
sn.relplot(y='MPG_H',x='Price',data=data_set)
plt.show()
sn.relplot(y='MPG_C',x='Price',data=data_set)
plt.show()
sn.relplot(y='Popularity',x='Price',data=data_set)
plt.show()

def calculate(y_test,y_pred):
    print(r2_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))
    print(mean_absolute_error(y_test, y_pred))

X = data_set.iloc[:,[8]].values
y = data_set.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
calculate(y_test,y_pred)

#knn
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
calculate(y_test,y_pred)





