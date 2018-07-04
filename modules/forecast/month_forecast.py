import os
import configparser
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import to_datetime
from pandas import date_range
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from numpy import array
from utils import *

forecast_folder = os.getcwd()


def get_data_frame(dir):
    df = read_csv(dir)
    df = df.rename(columns={'ef_date': 'date'})
    df.index = to_datetime(df.date, format='%Y%m%d')
    del df['date']
    return df


config = configparser.ConfigParser()
config.read(os.path.join(forecast_folder, 'configs', 'configs.ini'))
month = config['Month']
dataframe = get_data_frame('./data/store3.csv')

print(dataframe)


name_model_scaler = 'revenue_month_3_1'
path_model_scaler = './models/' + name_model_scaler

# preprocess data
dataframe = dataframe.groupby(level=0).sum()
dataframe = dataframe.sort_index().asfreq('d')
idx = date_range(dataframe.index.min().to_pydatetime().replace(day=1), '2017-11-30')
dataframe = dataframe.reindex(idx).fillna(method='backfill')
dataframe = dataframe.resample('M').sum()

scaler, train, test = prepare_data(DataFrame, MinMaxScaler, concat, dataframe, int(month['n_test']),
                                   int(month['n_lag']), int(month['n_seq']))

model = fit_lstm(Sequential, LSTM, Dense, train, int(month['n_lag']), int(month['n_seq']), int(month['n_batch']),
                 int(month['n_epochs']), int(month['n_neurons']))
# fit an LSTM network to training data
save_model_scaler(os, joblib, forecast_folder, model, scaler, name_model_scaler)
#
# model, scaler = load_model(os, forecast_folder, model_from_json, joblib, name_model_scaler)
#
# last_object = [-0.697399, -0.625974, -0.769114, -0.852086, -0.695439, -0.735706, -0.589509, 1.000000, -0.163192,
#                -0.485333, -0.582946, -0.538189]
# n_batch = 1
# result_forecast = make_forecasts(model, n_batch, test, int(month['n_lag']))
#
# forecasts = inverse_transform(array, result_forecast, scaler)
# print(forecasts)

# plot_forecasts(pyplot, dataframe, forecasts, int(month['n_test']), int(month['n_lag']), int(month['n_seq']))
