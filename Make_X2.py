import numpy as np
import pandas as pd
import pybithumb
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)

Scaler = MinMaxScaler()

home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)


def min_max_scaler(price):
    Scaler = MinMaxScaler()
    Scaler.fit(price)

    return Scaler.transform(price)


def low_high(Coin, input_data_length, trade_limit=0):

    #   거래 제한은 고점과 저점을 분리한다.

    #   User-Agent Configuration
    ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')

    price_gap = ohlcv_excel.close.max() / ohlcv_excel.close.min()
    if (price_gap < 1.07) and (trade_limit == 1):
        return None, None

    obv = [0] * len(ohlcv_excel)
    for m in range(1, len(ohlcv_excel)):
        if ohlcv_excel['close'].iloc[m] > ohlcv_excel['close'].iloc[m - 1]:
            obv[m] = obv[m - 1] + ohlcv_excel['volume'].iloc[m]
        elif ohlcv_excel['close'].iloc[m] == ohlcv_excel['close'].iloc[m - 1]:
            obv[m] = obv[m - 1]
        else:
            obv[m] = obv[m - 1] - ohlcv_excel['volume'].iloc[m]
    ohlcv_excel['OBV'] = obv

    closeprice = ohlcv_excel['close'].iloc[-1]

    # ----------- dataX, dataY 추출하기 -----------#
    #   OBV :
    ohlcv_data = ohlcv_excel.values[1:].astype(np.float)

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        price = ohlcv_data[:, :4]
        volume = ohlcv_data[:, [4]]
        OBV = ohlcv_data[:, [-1]]

        scaled_price = min_max_scaler(price)
        scaled_volume = min_max_scaler(volume)
        scaled_OBV = min_max_scaler(OBV)
        # print(scaled_MA60.shape)

        x = np.concatenate((scaled_price, scaled_OBV), axis=1)  # axis=1, 세로로 합친다

        if (x[-1][1] > 0.3) and (trade_limit == 1):
            return None, None

        # print(x.shape)  # (258, 6)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        for i in range(input_data_length, len(ohlcv_data) + 1):  # 마지막 데이터까지 다 긇어모은다.
            group_x = x[i - input_data_length:i]
            dataX.append(group_x)  # dataX 리스트에 추가

        if (len(dataX) < 100) and (trade_limit == 1):
            return None, None

        X_test = np.array(dataX)
        row = X_test.shape[1]
        col = X_test.shape[2]

        X_test = X_test.astype('float32').reshape(-1, row, col, 1)

        return X_test, closeprice


def made_x(file, input_data_length, model_num, check_span, get_fig):

    ohlcv_excel = pd.read_excel(dir + file, index_col=0)

    obv = [0] * len(ohlcv_excel)
    for m in range(1, len(ohlcv_excel)):
        if ohlcv_excel['close'].iloc[m] > ohlcv_excel['close'].iloc[m - 1]:
            obv[m] = obv[m - 1] + ohlcv_excel['volume'].iloc[m]
        elif ohlcv_excel['close'].iloc[m] == ohlcv_excel['close'].iloc[m - 1]:
            obv[m] = obv[m - 1]
        else:
            obv[m] = obv[m - 1] - ohlcv_excel['volume'].iloc[m]
    ohlcv_excel['OBV'] = obv

    #   이후 check_span 데이터와 현재 포인트를 비교해서 현재 포인트가 저가인지 고가인지 예측한다.
    #   최대 3개의 중복 값을 허용한다.
    #   고저점을 잡아주는 함수 구현
    list_low_check = [np.NaN] * len(ohlcv_excel)
    list_high_check = [np.NaN] * len(ohlcv_excel)
    for i in range(len(ohlcv_excel) - check_span):
        if ohlcv_excel['close'][i + 1:i + 1 + check_span].min() >= ohlcv_excel['close'][i]:
            if ohlcv_excel['close'][i:i + 1 + check_span].value_counts().sort_index().iloc[0] <= 3:
                list_low_check[i] = 1
            else:
                list_low_check[i] = 0
        else:
            list_low_check[i] = 0

        if ohlcv_excel['close'][i + 1:i + 1 + check_span].max() <= ohlcv_excel['close'][i]:
            if ohlcv_excel['close'][i:i + 1 + check_span].value_counts().sort_index().iloc[-1] <= 3:
                list_high_check[i] = 1
            else:
                list_high_check[i] = 0
        else:
            list_high_check[i] = 0

    ohlcv_excel['low_check'] = list_low_check
    ohlcv_excel['high_check'] = list_high_check

    # ----------- dataX, dataY 추출하기 -----------#
    # print(ohlcv_excel.info())
    # ohlcv_excel.to_excel('test.xlsx')
    # quit()

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    #   OBV : -CHECK_SPAN
    ohlcv_data = ohlcv_excel.values[1: -check_span].astype(np.float)
    # print(pd.DataFrame(ohlcv_data).info())
    # print(pd.DataFrame(ohlcv_data).to_excel('test.xlsx'))
    # print(list(map(float, ohlcv_data[0])))
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        price = ohlcv_data[:, :4]
        volume = ohlcv_data[:, [4]]
        OBV = ohlcv_data[:, [-3]]

        #   Flexible Y_data    #
        low_check = ohlcv_data[:, [-2]]
        high_check = ohlcv_data[:, [-1]]

        scaled_price = min_max_scaler(price)
        scaled_volume = min_max_scaler(volume)
        scaled_OBV = min_max_scaler(OBV)

        x = np.concatenate((scaled_price, scaled_volume, scaled_OBV), axis=1)  # axis=1, 세로로 합친다
        y_low = low_check
        y_high = high_check
        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY_low = []  # Target 을 담을 그릇
        dataY_high = []  # Target 을 담을 그릇

        for i in range(input_data_length, len(ohlcv_data)):
            # group_x >> 이전 완성된 데이터를 사용해보도록 한다. (진입하는 시점은 데이터가 완성되어있지 않으니까)
            group_x = x[i - input_data_length: i]  # group_y 보다 1개 이전 데이터
            group_y_low = y_low[i]
            group_y_high = y_high[i]
            # print(group_x.shape)  # (28, 6)
            # print(group_y.shape)  # (1,)
            # quit()
            # if i == len(y) - 1:
            #     # print(group_x, "->", group_y)
            #     print(group_x[-1])
            #     print(x[i - 1])
            #     quit()
            dataX.append(group_x)  # dataX 리스트에 추가
            dataY_low.append(group_y_low)  # dataY 리스트에 추가
            dataY_high.append(group_y_high)  # dataY 리스트에 추가

        if len(dataX) < 100:
            return None

        #       Exstracting fiexd X_data       #
        sliced_ohlcv = ohlcv_data[input_data_length:, :6]

        #                      Get Figure                     #
        if get_fig == 1:
            spanlist_low = []
            spanlist_high = []

            for m in range(len(low_check)):
                if low_check[m] > 0.5:
                    if m + 1 < len(low_check):
                        spanlist_low.append((m, m + 1))
                    else:
                        spanlist_low.append((m - 1, m))

            for m in range(len(high_check)):
                if high_check[m] > 0.5:
                    if m + 1 < len(high_check):
                        spanlist_high.append((m, m + 1))
                    else:
                        spanlist_high.append((m - 1, m))

            # ----------- 인덱스 초기화 됨 -----------#

            # ----------- 공통된 Chart 그리기 -----------#

            plt.subplot(211)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            plt.plot(scaled_OBV, 'b', label='MA60')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_low)):
                plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='m', alpha=0.5)

            plt.subplot(212)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            plt.plot(scaled_OBV, 'b', label='MA60')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_high)):
                plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='c', alpha=0.5)

            Date = file.split()[0]
            Coin = file.split()[1].split('.')[0]
            plt.savefig('./Figure_data/%s_%s/%s %s.png' % (input_data_length, model_num, Date, Coin), dpi=500)
            plt.close()
            # plt.show()
            # ----------- Chart 그리기 -----------#

        return dataX, dataY_low, dataY_high, sliced_ohlcv


if __name__ == '__main__':

    # ----------- Params -----------#
    input_data_length = 54
    model_num = input('Press model number : ')

    #       Make folder      #
    try:
        os.mkdir('./Figure_data/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass
    check_span = 30
    get_fig = 0

    Made_X = []
    Made_Y = []
    Made_Y_low = []
    Made_Y_high = []

    for file in ohlcv_list:

        if int(file.split()[0].split('-')[1]) == 1:
            continue

        # file = '2019-10-27 LAMB ohlcv.xlsx'

        result = made_x(file, input_data_length, model_num, check_span, get_fig)
        # result = low_high('FX', input_data_length)
        # quit()

        # ------------ 데이터가 있으면 dataX, dataY 병합하기 ------------#
        if result is not None:

            Made_X += result[0]
            Made_Y_low += result[1]
            Made_Y_high += result[2]

            # 누적 데이터량 표시
            print(file, len(Made_X))

    # SAVING X, Y
    X = np.array(Made_X)
    Y_low = np.array(Made_Y_low)
    Y_high = np.array(Made_Y_high)

    # np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), X)
    # np.save('./Made_X_low/Made_Y %s_%s' % (input_data_length, model_num), Y_low)
    # np.save('./Made_X_high/Made_Y %s_%s' % (input_data_length, model_num), Y_high)

