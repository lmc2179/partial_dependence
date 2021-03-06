#Run curl https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv > BostonHousing.csv
from partial_dependence_analysis import mean_partial_dependence
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('BostonHousing.csv')

covar_cols = ['crim', 'dis', 'rad', 'chas', 'ptratio', 'rm']
target_col = 'medv'

X = df[covar_cols]
y = df[target_col]

m = RandomForestRegressor(n_estimators=1000)
m.fit(X, y)

crim_range = np.linspace(X['crim'].min(), X['crim'].max(), 100)
avg_pd = mean_partial_dependence(m, X, y, 'crim', crim_range)

[plt.axvline(q, linestyle='dotted') for q in X['crim'].quantile([0, 0.25, 0.5, 0.75, 1.0])]
plt.plot(crim_range, avg_pd)
plt.show() 
