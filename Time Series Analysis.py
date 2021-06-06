import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

electric = pd.read_csv("electric.csv")

# Look the time series dataset
print(electric.head())

# we make date columns treated as time
electric = pd.read_csv("electric.csv",parse_dates=['DATE'])
print(electric.head())

# index date as time to perform time series analysis
electric = pd.read_csv("electric.csv",parse_dates=['DATE'],index_col=['DATE'])
print(electric.head())

# see the records 01-01-1985 to 01-04-1986

print(electric['1985-01-01':'1986-04-01'])

plt.plot(electric)
plt.show()

from pylab import rcParams
rcParams['figure.figsize']=12,5
plt.plot(electric)
plt.show()

electric_log = electric.copy()

import numpy as np

electric_log = np.log(electric)

print(electric_log.Value)

# make a log transform model 

electric_mul_decompose = seasonal_decompose(electric, model='multiplicative')
electric_mul_decompose.plot()
plt.show()

#lets have a glance of Original & Log Transform Time Series !
plt.subplot(2,1,1)
plt.title('Original Time Series')
plt.plot(electric)

plt.subplot(2,1,2)
plt.title("Log Transform Time Series")
plt.plot(electric_log)
plt.show()