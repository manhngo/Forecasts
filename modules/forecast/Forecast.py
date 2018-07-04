class Forecast:
    import os
    import configparser
    from pandas import DataFrame
    from pandas import Series
    from pandas import concat
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
    # convert time series into supervised learning problem

    def __init__(self, forecast_folder, type_name, name_forecast):
        self.model = None
        self.scaler = None
        self.train = self.array([])
        self.test = self.array([])
        self.forecast_values = []
        self.forecast_folder = forecast_folder
        config = self.configparser.ConfigParser()
        config.read(self.os.path.join(self.forecast_folder, 'configs', 'configs.ini'))
        type = config[type_name]
        self.n_lag = type['n_lag']
        self.n_seq = type['n_seq']
        self.n_test = type['n_test']
        self.n_epochs = type['n_epochs']
        self.n_batch = type['n_batch']
        self.n_neurons = type['n_neurons']
        self.name_forecast = name_forecast

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = self.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = self.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # transform series into train and test sets for supervised learning
    def prepare_data(self, series):
        # extract raw values
        raw_values = series.values
        # transform data to be stationary
        # diff_series = difference(raw_values, 1)
        # diff_values = diff_series.values
        diff_values = raw_values.reshape(len(raw_values), 1)
        # rescale values to -1, 1
        self.scaler = self.MinMaxScaler(feature_range=(-1, 1))
        scaled_values = self.scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        # transform into supervised learning problem X, y
        supervised = self.series_to_supervised(scaled_values, self.n_lag, self.n_seq)
        print(supervised)
        supervised_values = supervised.values
        # split into train and test sets
        self.train, self.test = supervised_values[0:-self.n_test], supervised_values[-self.n_test:]

    def fit_lstm(self):
        # reshape training into [samples, timesteps, features]
        X, y = self.train[:, 0: self.n_lag], self.train[:, self.n_lag:]
        print(X, y)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        # design network
        self.model = self.Sequential()
        self.model.add(self.LSTM(self.n_neurons, batch_input_shape=(self.n_batch, X.shape[1], X.shape[2]), stateful=True))
        self.model.add(self.Dense(y.shape[1]))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        # fit network
        for i in range(self.n_epochs):
            print('n', i)
            self.model.fit(X, y, epochs=1, batch_size=self.n_batch, verbose=2, shuffle=False)
            self.model.reset_states()
        return self.model

    def save_model_scaler(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.os.path.join(self.forecast_folder, "models", self.name_forecast + '.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
            self.model.save_weights(self.os.path.join(self.forecast_folder, "models", self.name_forecast + '.h5'))
        self.joblib.dump(self.scaler, self.os.path.join(self.forecast_folder, "models", self.name_forecast + '.pkl'))
        print("Saved model to disk")

    def load_model(self):
        model_str = self.os.path.join(self.forecast_folder, 'models', self.name_forecast + '.json')
        weight_str = self.os.path.join(self.forecast_folder, 'models', self.name_forecast + '.h5')
        scaler_str = self.os.path.join(self.forecast_folder, 'models', self.name_forecast + '.pkl')
        json_file = open(model_str, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = self.model_from_json(loaded_model_json)
        loaded_model.load_weights(weight_str)
        scaler = self.joblib.load(scaler_str)
        return loaded_model, scaler

    # make one forecast with an LSTM,
    def forecast_lstm(self, X, n_batch):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = self.model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]

    # evaluate the persistence model
    def make_forecasts(self, n_batch, test):
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:self.n_lag], test[i, self.n_lag:]
            # make forecast
            forecast = self.forecast_lstm(X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts

    # inverse data transform on forecasts
    def inverse_transform(self, forecasts, scaler):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = self.array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            inverted.append(inv_scale)
        return inverted

    def plot_forecasts(self, series):
        # plot the entire dataset in blue
        self.pyplot.plot(series.values)
        # plot the forecasts in red
        for i in range(len(self.forecast_values)):
            off_s = len(series) - self.n_lag - self.n_seq + 1 - self.n_test + i
            print(off_s)
            off_e = off_s + len(self.forecast_values[i]) + 1
            xaxis = [x for x in range(off_s, off_e)]
            print(xaxis)
            yaxis = [series.values[off_s]] + self.forecast_values[i].tolist()
            print(yaxis)
            self.pyplot.plot(xaxis, yaxis, color='red')
            # show the plot
            self.pyplot.show()

    def evaluate_forecasts(self):
        for i in range(self.n_seq):
            actual = [row[i] for row in self.test]
            predicted = [forecast[i] for forecast in self.forecast_values]
            rmse = self.sqrt(self.mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i + 1), rmse))
