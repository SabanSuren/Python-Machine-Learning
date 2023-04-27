# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("positions.csv")
level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values

regression = RandomForestRegressor(n_estimators=10, random_state=2)
regression.fit(level,salary)

print(regression.predict(8.3))
