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


def low_high(Coin, input_data_length):

    #   Proxy 설정 해주기
    ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')

    closeprice = ohlcv_excel['close'].iloc[-1]
    ohlcv_excel['MA60'] = ohlcv_excel['close'].rolling(60).mean()

    # ----------- dataX, dataY 추출하기 -----------#
    ohlcv_data = ohlcv_excel.values[ohlcv_excel['MA60'].isnull().sum():].astype(np.float)

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        price = ohlcv_data[:, :4]
        volume = ohlcv_data[:, [4]]
        MA60 = ohlcv_data[:, [-1]]

        scaled_price = min_max_scaler(price)
        scaled_volume = min_max_scaler(volume)
        scaled_MA60 = min_max_scaler(MA60)
        # print(scaled_MA60.shape)

        x = np.concatenate((scaled_price, scaled_volume, scaled_MA60), axis=1)  # axis=1, 세로로 합친다
        # print(x.shape, y.shape)  # (258, 6) (258, 1)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        for i in range(input_data_length, len(ohlcv_data) + 1):  # 마지막 데이터까지 다 긇어모은다.
            group_x = x[i - input_data_length:i]
            dataX.append(group_x)  # dataX 리스트에 추가

        if len(dataX) < 100:
            return None, None

        X_test = np.array(dataX)
        row = X_test.shape[1]
        col = X_test.shape[2]

        X_test = X_test.astype('float32').reshape(-1, row, col, 1)

        return X_test, closeprice


def made_x(file, input_data_length, Range_fluc, check_span, get_fig):

    ohlcv_excel = pd.read_excel(dir + file, index_col=0)

    # period = 9
    # ohlcv_excel['closegap_cunsum'] = (ohlcv_excel['close'] - ohlcv_excel['close'].shift(1)).cumsum()
    # ohlcv_excel['closegap_abs_cumsum'] = abs(ohlcv_excel['close'] - ohlcv_excel['close'].shift(1)).cumsum()
    # # print(ohlcv_excel)
    #
    # ohlcv_excel['CMO'] = (ohlcv_excel['closegap_cunsum'] - ohlcv_excel['closegap_cunsum'].shift(period)) / (
    #         ohlcv_excel['closegap_abs_cumsum'] - ohlcv_excel['closegap_abs_cumsum'].shift(period)) * 100
    #
    # del ohlcv_excel['closegap_cunsum']
    # del ohlcv_excel['closegap_abs_cumsum']

    ohlcv_excel['MA60'] = ohlcv_excel['close'].rolling(60).mean()
    # 이전 & 이후 check_span 데이터와 현재 포인트를 비교해서 현재 포인트가 저가인지 고가인지 예측한다.
    ohlcv_excel['fluc_close'] = ohlcv_excel['close'].shift(-10).rolling(10).max() / ohlcv_excel['close'].shift(1)
    ohlcv_excel['low_check'] = np.where((ohlcv_excel['close'].shift(1).rolling(check_span).min() > ohlcv_excel['close'])
                                        & (ohlcv_excel['close'].shift(-check_span).rolling(check_span).min() >
                                           ohlcv_excel['close']), 1, 0)
    ohlcv_excel['high_check'] = np.where((ohlcv_excel['close'].shift(1).rolling(check_span).max() < ohlcv_excel['close'])
                                         & (ohlcv_excel['close'].shift(-check_span).rolling(check_span).max() <
                                            ohlcv_excel['close']), 1, 0)

    # ----------- dataX, dataY 추출하기 -----------#
    # print(ohlcv_excel.info())
    # ohlcv_excel.to_excel('test.xlsx')
    # quit()

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    # MA60 부터 FLUC_CLOSE, 존재하는 값만 슬라이싱
    if check_span < 60:
        ohlcv_data = ohlcv_excel.values[ohlcv_excel['MA60'].isnull().sum(): -check_span].astype(np.float)
    else:
        ohlcv_data = ohlcv_excel.values[check_span: -check_span].astype(np.float)
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
        # CMO = ohlcv_data[:, [-5]]
        MA60 = ohlcv_data[:, [-4]]

        #   Flexible Y_data    #
        fluc_close = ohlcv_data[:, [-3]]
        low_check = ohlcv_data[:, [-2]]
        high_check = ohlcv_data[:, [-1]]

        scaled_price = min_max_scaler(price)
        scaled_volume = min_max_scaler(volume)
        # scaled_CMO = min_max_scaler(CMO)
        scaled_MA60 = min_max_scaler(MA60)
        # print(scaled_MA60.shape)

        fluc_close = np.array(list(map(lambda x: 1 if x > Range_fluc else 0, fluc_close)))
        fluc_close = fluc_close.reshape(-1, 1)
        # print(fluc_close.shape)

        x = np.concatenate((scaled_price, scaled_volume, scaled_MA60), axis=1)  # axis=1, 세로로 합친다
        y = fluc_close
        y_low = low_check
        y_high = high_check
        # print(x.shape, y.shape)  # (258, 6) (258, 1)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇
        dataY_low = []  # Target 을 담을 그릇
        dataY_high = []  # Target 을 담을 그릇

        for i in range(input_data_length, len(y)):
            # group_x >> 이전 완성된 데이터를 사용해보도록 한다. (진입하는 시점은 데이터가 완성되어있지 않으니까)
            group_x = x[i - input_data_length: i]  # group_y 보다 1개 이전 데이터
            group_y = y[i]  # i = len(y) - 1
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
            dataY.append(group_y)  # dataY 리스트에 추가
            dataY_low.append(group_y_low)  # dataY 리스트에 추가
            dataY_high.append(group_y_high)  # dataY 리스트에 추가

        #       Exstracting fiexd X_data       #
        sliced_ohlcv = ohlcv_data[input_data_length:, :6]

        # ----------- FLUC_CLOSE TO SPAN, 넘겨주기 위해서 INDEX 를 담아주어야 한다. -----------#
        if get_fig == 1:
            spanlist = []
            spanlist_low = []
            spanlist_high = []
            raw_fluc_close = ohlcv_data[:, [-3]]
            for m in range(len(raw_fluc_close)):
                if raw_fluc_close[m] > Range_fluc:
                    if m + 1 < len(raw_fluc_close):
                        spanlist.append((m, m + 1))
                    else:
                        spanlist.append((m - 1, m))

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
            plt.subplot(311)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            plt.plot(scaled_MA60, 'b', label='MA60')
            plt.legend(loc='upper right')

            # Spanning
            for i in range(len(spanlist)):
                plt.axvspan(spanlist[i][0], spanlist[i][1], facecolor='g', alpha=0.5)
            plt.subplot(312)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            plt.plot(scaled_MA60, 'b', label='MA60')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_low)):
                plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='m', alpha=0.5)

            plt.subplot(313)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            plt.plot(scaled_MA60, 'b', label='MA60')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_high)):
                plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='c', alpha=0.5)

            Date = file.split()[0]
            Coin = file.split()[1].split('.')[0]
            plt.savefig('./Figure_data/%s %s.png' % (Date, Coin), dpi=500)
            plt.close()
            # plt.show()
            # ----------- Chart 그리기 -----------#

        return dataX, dataY, dataY_low, dataY_high, sliced_ohlcv


if __name__ == '__main__':

    # ----------- Params -----------#
    input_data_length = 54
    Range_fluc = 1.035  # >> Best Param 을 찾도록 한다.
    check_span = 40
    get_fig = 0

    Made_X = []
    Made_Y = []
    Made_Y_low = []
    Made_Y_high = []

    for file in ohlcv_list:

        # file = '2019-10-21 BCD ohlcv.xlsx'

        result = made_x(file, input_data_length, Range_fluc, check_span, get_fig)

        # ------------ 데이터가 있으면 dataX, dataY 병합하기 ------------#
        if result is not None:

            Made_X += result[0]
            Made_Y += result[1]
            Made_Y_low += result[2]
            Made_Y_high += result[3]

            # 누적 데이터량 표시
            print(file, len(Made_X))  # 현재까지 321927개
            if len(Made_X) > 300000:
                break

    # SAVING X, Y
    X = np.array(Made_X)
    Y = np.array(Made_Y)
    Y_low = np.array(Made_Y_low)
    Y_high = np.array(Made_Y_high)
    print(np.sum(Y))

    np.save('./Made_X/Made_X %s' % input_data_length, X)
    # np.save('./Made_X/Made_Y %s' % input_data_length, Y)
    np.save('./Made_X_low/Made_Y %s' % input_data_length, Y_low)
    np.save('./Made_X_high/Made_Y %s' % input_data_length, Y_high)

    plt.plot(Y)
    plt.savefig('./Made_X/Made_Y %s.png' % input_data_length)
    plt.close()
