# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv("hw_25000.csv")


boy = data.Height.values.reshape(-1,1)
kilo =data.iloc[:,[0,1]].values

regression = LinearRegression()
regression.fit(boy,kilo)

print(regression.predict(np.array([[40,120]])))
# print(regression.predict(np.array[[62]]))
# print(regression.predict(np.array[[64]]))
# print(regression.predict(np.array[[66]]))
# print(regression.predict(np.array[[66]]))
# print(regression.predict(np.array[[68]]))


print(data.columns)

plt.scatter(data.Height,data.Weight)
x = np.arange(min(data.Height),max(data.Height)).reshape(-1,1)
plt.plot(x,regression.predict(x),color="red")
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("Simple Linear Regression Model")
plt.show()

print(r2_score(kilo,regression.predict(boy)))










