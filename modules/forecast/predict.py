import numpy
from pandas import read_csv
from keras.models import model_from_json
import os
from django.db import connection
from keras.backend import clear_session
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from django.conf import settings

cursor = connection.cursor()


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility
numpy.random.seed(7)


def make_predict(id_store, type):
    forecast_folder = os.path.join(settings.BASE_DIR, "modules", "forecast")
    data_file = os.path.join(forecast_folder, 'data', str(id_store) + '.csv')
    data_frame = read_csv(data_file, usecols=[1], engine='python', skipfooter=0)
    data_set = data_frame.values
    data_set = data_set.astype('float32')
    cont = numpy.amax(data_set)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_set = scaler.fit_transform(data_set)
    train_size = int(len(data_set) - 32)
    test_size = len(data_set) - train_size
    train, test = data_set[0:train_size, :], data_set[train_size:len(data_set), :]
    look_back = 1
    testX, testY = create_dataset(test, look_back)
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    test1 = testX[0:3]
    train1 = numpy.sum(testX)
    train1 = train1 * cont
    data = testX[:]
    model_str = os.path.join(forecast_folder, 'models', 'model_' + str(id_store) + '.json')
    json_file = open(model_str, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    model_str_1 = os.path.join(forecast_folder, 'models', 'model_' + str(id_store) + '.h5')
    loaded_model.load_weights(model_str_1)
    period = 30
    if type == "1":
        period = 30
    if type == "2":
        period = 120
    if type == "3":
        period = 360
    for i in range(period):
        test_predict = loaded_model.predict(test1)
        print('1', test_predict)
        test_predict = numpy.reshape(test_predict, (test_predict.shape[0], 1, test_predict.shape[1]))
        test2 = test_predict[len(test_predict) - 2:len(test_predict) - 1]
        test1 = numpy.append(test1, test2, axis=0)
    test1 = test1[3:3 + period]
    predict = numpy.sum(test1)
    predict = predict * cont
    clear_session()
    result = []
    result1 = []
    str1 = str(int(predict))
    str2 = str(int(train1))
    str3, str4 = "", ""
    len1 = len(str1)
    len2 = len(str2)
    len3 = len1 % 3
    if len3 == 0:
        len3 = 3
    len4 = len2 % 3
    if len4 == 0:
        len4 = 3
    for i in range(len1):
        if i == len3 - 1 or i % 3 == len3 - 1:

            str3 = str3 + str1[i]
            str3 = str3 + '.'
        else:

            str3 = str3 + str1[i]
    for i in range(len2):
        if i == len4 - 1 or i % 3 == len4 - 1:
            str4 = str4 + str2[i]
            str4 = str4 + '.'
        else:
            str4 = str4 + str2[i]
    print("doanh thu du doan trong thang tiep theo: %2f" % (predict))
    print("doanh thu thuc te thang do thu duoc : %2f " % (train1))

    result.append(str3)
    result1.append(str3)
    result.append(str4)
    if (train1 < predict):
        str3 = str(round(((train1 / predict) * 100), 2))
        result.append(str3)
        print("ty le chinh xac : %2f%%" % ((train1 / predict) * 100))

    else:
        str3 = str(round(((predict / train1) * 100), 2))
        result.append(str3)
        print("ty le chinh xac : %2f%%" % ((predict / train1) * 100))
    if (type == "1"):
        return result
    else:
        return result1


def main():
    file = open("aaa1.csv", "w")
    file.write('"ma cua hang","ten cua hang","doanh thu thuc te",ty le chinh xac"\n')
    for i in range(1, 91):
        query = ("SELECT store_name FROM sales_family_mart.store where store_id={}".format(i))
        cursor.execute(query)
        store_name = ""
        data = cursor.fetchall()
        for item in data:
            store_name = item[0]
        result = make_predict(i)
        file.write('"' + str(i) + '","' + store_name.encode('utf-8') + '","' + str(result[0]) + '","' + str(
            result[1]) + '","' + str(result[2]) + '"\n')
    for i in range(92, 149):
        query = ("SELECT store_name FROM sales_family_mart.store where store_id={}".format(i))
        cursor.execute(query)
        store_name = ""
        data = cursor.fetchall()
        for item in data:
            store_name = item[0]

        result = make_predict(i)
        file.write('"' + str(i) + '","' + store_name.encode('utf-8') + '","' + str(result[0]) + '","' + str(
            result[1]) + '","' + str(result[2]) + '"\n')
    # my code here


if __name__ == "__main__":
    main()
