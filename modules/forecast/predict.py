import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pandas import read_csv, to_datetime
from numpy import array
from keras.models import model_from_json
from keras.backend import clear_session
from django.conf import settings
from sklearn.externals import joblib

forecast_folder = os.path.join(settings.BASE_DIR, "modules", "forecast")
# forecast_folder = os.getcwd()

def get_data_frame(dir):
    df = read_csv(dir)
    df = df.rename(columns={'ef_date': 'date'})
    df.index = to_datetime(df.date, format='%Y%m%d')
    del df['date']
    return df


def load_model(name):
    model_str = os.path.join(forecast_folder, 'models', name + '.json')
    weight_str = os.path.join(forecast_folder, 'models', name + '.h5')
    scaler_str = os.path.join(forecast_folder, 'models', name + '.pkl')
    json_file = open(model_str, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_str)
    scaler = joblib.load(scaler_str)
    return loaded_model, scaler
# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]


def forecast(last_object, name):
    model, scaler = load_model(name)
    last_object = array([last_object])
    n_batch = 1
    result_forecast = forecast_lstm(model, last_object, n_batch)
    clear_session()
    result_forecast = array(result_forecast)
    result_forecast = result_forecast.reshape(1, len(result_forecast))

    inv_scale = scaler.inverse_transform(result_forecast)
    inv_scale = inv_scale.reshape(inv_scale.shape[1])
    return inv_scale


# if __name__ == "__main__":
#     print(forecast(179375488, 'revenue_1_1'))
