# create arima statsmodels
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import scipy.io as sio
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.io
from math import log
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import chi2
import plotly.tools as tls
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.plotly as py


look_back = 60
# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back):
	dataX,dataY = [],[]
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), :]
                dataX.append(a)
		dataY.append(dataset[i + look_back, :])
	return numpy.array(dataX), numpy.array(dataY)

def series_diff(series):
	first = [series[i+1] - series[i]   for i in range(len(series)- 1)]
        return numpy.asarray(first)

def series_log(series):
        series[series < 0.00000000001] = 0.00001
        l = list()
        g = numpy.log(series)
        return numpy.asarray(g)

        lags = range(2,100)
def hurst_ernie_chan(p):

    variancetau = []; tau = []

    for lag in lags:

        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)

        # Compute the log returns on all days, then compute the variance on the difference in log returns
        # call this pp or the price difference
        pp = subtract(p[lag:], p[:-lag])
        variancetau.append(var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    #print tau
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = polyfit(log10(tau),log10(variancetau),1)

    hurst = m[0] / 2

    return hurst



# fix random seed
numpy.random.seed(7)
#Chosing Dataset to be of 14 variables
rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 14)
# load the dataset
mat = scipy.io.loadmat('dock_vg_capped.mat')
dataset = mat['dock_avg']
dataset = series_log(dataset)
dataset = series_diff(dataset)
dataset1 = dataset
train_data=dataset[0:train_size,:]
dataset = dataset.astype('float')
scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = SelectKBest(chi2, k=2).fit_transform(dataset, y)
#dm=len(dataframe.columns)
#predict = dataset[ :,0]
fit = rfe.fit(dataset, predict)
ds_short = list()
trnsp_ds = numpy.transpose(dataset)
for i in range(len(fit.support_)):
	if fit.support_[i]:
		ds_short.append(trnsp_ds[i])
dataset = numpy.transpose(ds_short)
train_size=60500

dataset = scaler.fit_transform(train_data)
#trainX, trainY = create_dataset(train_data, look_back)
numpy.random.seed(7)



#build_model = model.fit(15)
for i in range(1,20):
	lag_order = 5 * i
		for p in [5*i for i in (range(20))]:
			for q in  [5*j for j in range(20)]:
				#results = model.fit(maxlags = lags, ic = typ)
				mod = sm.tsa.VARMAX(dataset, order=(p,q))
				res = mod.fit(maxiter=1000, disp=False)
				predictions = res.forecast(train_data[-lag_order:], 60)
				predicted_usage = predictions[:,0]
				fig = plt.figure(i)

				lag_order = results.k_ar
				predictions = results.forecast(train_data[-lag_order:], 60)
				predicted_usage = predictions[:,0]
				actual_usage = dataset1[60500:60560, 0]
				rms = sqrt(mean_squared_error(actual_usage, predicted_usage))
				ax = [i for i in range(60)]
				ac_mean = numpy.mean(actual_usage)
				pr_mean = numpy.mean(predicted_usage)
				predicted_usage = numpy.asarray([val - pr_mean for val in predicted_usage])
				actual_usage = numpy.asarray([val - ac_mean for val in actual_usage])
				plt.plot(actual_usage)
				plt.plot(predicted_usage)
				#plt.legend(['actual usage lags = '+ str(lags) , 'Predicted usage, rms = ' + str(rms) ], loc = 'upper left')
				fig.savefig('p='+str(p)+',q=' +str(q) +",rms="+str(rms)+ " .png")
				fig.clear()
