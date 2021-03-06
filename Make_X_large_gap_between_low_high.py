import numpy as np
import pandas as pd
import pybithumb
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 1500)

Scaler = MinMaxScaler()

home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)


def min_max_scaler(price):
    Scaler = MinMaxScaler()
    Scaler.fit(price)

    return Scaler.transform(price)


def low_high(Coin, input_data_length, ip_limit=None, trade_limit=None, crop=None, sudden_death=0):

    #   거래 제한은 고점과 저점을 분리한다.

    #   User-Agent Configuration
    #   IP - Change
    if ip_limit is None:
        ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
    else:
        ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1', 'proxyon')

    price_gap = ohlcv_excel.close.max() / ohlcv_excel.close.min()
    if (price_gap < 1.07) and (trade_limit is not None):
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

        #       OBV 가 급격히 떨어져서 진입 / 이탈 예측이 어려워진 경우      #
        if crop is not None:
            crop_size = 300
            if len(scaled_OBV) > crop_size:
                if sum(scaled_OBV[-crop_size:] < 0.2) >= crop_size - 50:  # sum 의 결과값은 list 이지만 == crop_size 와 비교가능하다.
                    if scaled_OBV[-crop_size] < 0.2:  # 맨 앞 OBV 가 0.2 보다 작으면,
                        scaled_price = min_max_scaler(price[-crop_size:])
                        scaled_volume = min_max_scaler(volume[-crop_size:])
                        scaled_OBV = min_max_scaler(OBV[-crop_size:])

        x = np.concatenate((scaled_price, scaled_volume, scaled_OBV), axis=1) + sudden_death  # axis=1, 세로로 합친다

        if (x[-1][1] > 0.3) and (trade_limit is not None):
            return None, None

        # print(x.shape)  # (258, 6)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        for i in range(input_data_length, len(x) + 1):  # 마지막 데이터까지 다 긇어모은다.
            group_x = x[i - input_data_length:i]
            dataX.append(group_x)  # dataX 리스트에 추가

        if (len(dataX) < 100) and (trade_limit is not None):
            return None, None

        X_test = np.array(dataX)

        row = X_test.shape[1]
        col = X_test.shape[2]

        X_test = X_test.astype('float32').reshape(-1, row, col, 1)

        return X_test, closeprice


def made_x(file, input_data_length, model_num, ascend_gap, check_span, get_fig):

    ohlcv_excel = pd.read_excel(dir + file, index_col=0)

    # ----------- dataX, dataY 추출하기 -----------#
    # print(ohlcv_excel.info())
    # ohlcv_excel.to_excel('test.xlsx')
    # quit()

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    #   OBV : -CHECK_SPAN
    ohlcv_data = ohlcv_excel.values[:].astype(np.float)
    # ohlcv_data = ohlcv_excel.values[1: -check_span].astype(np.float)
    # print(pd.DataFrame(ohlcv_data).info())
    # print(pd.DataFrame(ohlcv_data).to_excel('test.xlsx'))
    # print(list(map(float, ohlcv_data[0])))
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        price = ohlcv_data[:, :4]
        # volume = ohlcv_data[:, [4]]
        # OBV = ohlcv_data[:, [-2]]

        scaled_price = min_max_scaler(price)
        # scaled_volume = min_max_scaler(volume)
        # scaled_OBV = min_max_scaler(OBV)

        closeprice = scaled_price[:, [1]]
        trade_state = [np.NaN] * len(ohlcv_data)
        for i in range(check_span, len(ohlcv_data) - check_span):
            #   저점이면서 급상승을 예측하는
            if closeprice[i + 1:i + 1 + check_span].min() >= closeprice[i]:
                if closeprice[i + 1:i + 1 + check_span].max() - closeprice[i] >= ascend_gap:
                    trade_state[i] = 1
                else:
                    trade_state[i] = 0
            #   급상승된 차트의 꼭대기 부근
            elif closeprice[i + 1:i + 1 + check_span].max() <= closeprice[i]:
                if closeprice[i] - closeprice[i - check_span:i].min() >= ascend_gap:
                    trade_state[i] = 2
                else:
                    trade_state[i] = 0
            #   거래 안함
            else:
                trade_state[i] = 0

        #   Flexible Y_data    #
        # print(trade_state)
        trade_state = np.array(trade_state).reshape(-1, 1).astype(np.float)
        # print(len(ohlcv_data) - np.isnan(trade_state).sum())
        # quit()

        # x = np.concatenate((scaled_price, scaled_volume, scaled_OBV), axis=1)  # axis=1, 세로로 합친다
        x = scaled_price[check_span:len(ohlcv_data) - check_span]
        y = trade_state[check_span:len(ohlcv_data) - check_span]
        # print(len(y))
        # quit()
        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇

        for i in range(input_data_length, len(y)):
            # group_x >> 이전 완성된 데이터를 사용해보도록 한다. (진입하는 시점은 데이터가 완성되어있지 않으니까)
            group_x = x[i - input_data_length: i]  # group_y 보다 1개 이전 데이터
            group_y = y[i]
            # print(group_x.shape)  # (28, 6)
            # print(group_y.shape)  # (1,)
            # quit()
            # if i == len(y) - 1:
            #     # print(group_x, "->", group_y)
            #     print(group_x[-1])
            #     print(x[i - 1])
            #     quit()
            dataX.append(group_x)  # dataX 리스트에 추가
            dataY.append(group_y)

        if len(dataX) < 100:
            return None

        #       Exstracting fiexd X_data       #
        sliced_ohlcv = ohlcv_data[input_data_length:, :6]

        #                      Get Figure                     #
        if get_fig == 1:
            spanlist_low = []
            spanlist_high = []

            for m in range(len(trade_state)):
                if (trade_state[m] > 0.5) and (trade_state[m] < 1.5):
                    if m + 1 < len(trade_state):
                        spanlist_low.append((m, m + 1))
                    else:
                        spanlist_low.append((m - 1, m))

            for m in range(len(trade_state)):
                if (trade_state[m] > 1.5) and (trade_state[m] < 2.5):
                    if m + 1 < len(trade_state):
                        spanlist_high.append((m, m + 1))
                    else:
                        spanlist_high.append((m - 1, m))

            # ----------- 인덱스 초기화 됨 -----------#

            # ----------- 공통된 Chart 그리기 -----------#

            plt.subplot(211)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'gold', label='close')
            # plt.plot(scaled_OBV, 'b', label='OBV')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_low)):
                plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='c', alpha=0.7)

            plt.subplot(212)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'gold', label='close')
            # plt.plot(scaled_OBV, 'b', label='OBV')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_high)):
                plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='m', alpha=0.7)

            Date = file.split()[0]
            Coin = file.split()[1].split('.')[0]
            plt.savefig('./Figure_data/%s_%s/%s %s.png' % (input_data_length, model_num, Date, Coin), dpi=500)
            plt.close()
            # plt.show()
            # ----------- Chart 그리기 -----------#

        return dataX, dataY, sliced_ohlcv


if __name__ == '__main__':

    # ----------- Params -----------#
    input_data_length = 54
    model_num = 35
    ascend_gap = 0.5
    check_span = 50
    get_fig = 1

    #       Make folder      #
    try:
        os.mkdir('./Figure_data/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass

    Made_X = []
    Made_Y = []
    # ohlcv_list = ['2020-01-12 ADA ohlcv.xlsx']

    for file in ohlcv_list:

        if int(file.split()[0].split('-')[1]) != 1:
            continue

        result = made_x(file, input_data_length, model_num, ascend_gap, check_span, get_fig)
        # result = low_high('FX', input_data_length)
        # quit()

        # ------------ 데이터가 있으면 dataX, dataY 병합하기 ------------#
        if result is not None:

            Made_X += result[0]
            Made_Y += result[1]

            # 누적 데이터량 표시
            print(file, len(Made_X))

    # SAVING X, Y
    X = np.array(Made_X)
    Y = np.array(Made_Y)

    np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), X)
    np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), Y)

