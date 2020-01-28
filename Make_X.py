import numpy as np
import pandas as pd
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


def made_x(file, input_data_length, Range_fluc, get_fig):

    ohlcv_excel = pd.read_excel(dir + file, index_col=0)
    ohlcv_excel['MA60'] = ohlcv_excel['close'].rolling(60).mean()
    # 이후 10개 CLOSE 데이터의 MAX / 현재 POINT 의 이전 CLOSE
    ohlcv_excel['fluc_close'] = ohlcv_excel['close'].shift(-10).rolling(10).max() / ohlcv_excel['close'].shift(1)

    # ----------- dataX, dataY 추출하기 -----------#
    # print(ohlcv_excel.tail(20))
    # quit()

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    # MA60 부터 FLUC_CLOSE, 존재하는 값만 슬라이싱
    ohlcv_data = ohlcv_excel.values[ohlcv_excel['MA60'].isnull().sum(): -ohlcv_excel['fluc_close'].isnull().sum()].astype(np.float)
    # print(pd.DataFrame(ohlcv_data).info())
    # print(list(map(float, ohlcv_data[0])))
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        # ----- 데이터 전처리 -----#
        price = ohlcv_data[:, :4]
        volume = ohlcv_data[:, [4]]
        MA60 = ohlcv_data[:, [-2]]
        fluc_close = ohlcv_data[:, [-1]]

        scaled_price = min_max_scaler(price)
        scaled_volume = min_max_scaler(volume)
        scaled_MA60 = min_max_scaler(MA60)
        # print(scaled_MA60.shape)

        fluc_close = np.array(list(map(lambda x: 1 if x > Range_fluc else 0, fluc_close)))
        fluc_close = fluc_close.reshape(-1, 1)
        # print(fluc_close.shape)

        x = np.concatenate((scaled_price, scaled_volume, scaled_MA60), axis=1)  # axis=1, 세로로 합친다
        y = fluc_close
        # print(x.shape, y.shape)  # (258, 6) (258, 1)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇

        for i in range(input_data_length + 1, len(y)):
            # group_x >> 이전 완성된 데이터를 사용해보도록 한다. (진입하는 시점은 데이터가 완성되어있지 않으니까)
            group_x = x[i - 1 - input_data_length: i - 1]  # 0 부터 len(y) - 2 까지 대입
            group_y = y[i]
            # print(group_x.shape)  # (28, 6)
            # print(group_y.shape)  # (1,)
            # quit()
            # if i is input_data_length + 1:
            #     print(group_x, "->", group_y)
            #     quit()
            dataX.append(group_x)  # dataX 리스트에 추가
            dataY.append(group_y)  # dataY 리스트에 추가

        sliced_ohlcv = ohlcv_data[input_data_length + 1:, :-1]

        # ----------- FLUC_CLOSE TO SPAN, 넘겨주기 위해서 INDEX 를 담아주어야 한다. -----------#
        if get_fig == 1:
            spanlist = []
            for m in range(len(ohlcv_data[:, [-1]])):
                if ohlcv_data[:, [-1]][m] > Range_fluc:
                    if m + 1 < len(ohlcv_data[:, [-1]]):
                        spanlist.append((m, m + 1))
                    else:
                        spanlist.append((m - 1, m))

            # ----------- 인덱스 초기화 됨 -----------#

            # ----------- Chart 그리기 -----------#
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            plt.plot(scaled_MA60, 'b', label='MA60')
            plt.legend(loc='upper right')

            # Spanning
            for i in range(len(spanlist)):
                plt.axvspan(spanlist[i][0], spanlist[i][1], facecolor='blue', alpha=0.3)

            Date = file.split()[0]
            Coin = file.split()[1].split('.')[0]
            plt.savefig('./Figure_data/%s %s.png' % (Date, Coin), dpi=500)
            plt.close()
            # plt.show()
            # ----------- Chart 그리기 -----------#

        return dataX, dataY, sliced_ohlcv


if __name__ == '__main__':

    for i in range(3, 4):

        # ----------- Params -----------#
        input_data_length = 6 * (i ** 2)
        Range_fluc = 1.035  # >> Best Param 을 찾도록 한다.
        get_fig = 0

        Made_X = []
        Made_Y = []

        for file in ohlcv_list:

            result = made_x(file, input_data_length, Range_fluc, get_fig)

            # ------------ 데이터가 있으면 dataX, dataY 병합하기 ------------#
            if result is not None:

                Made_X += result[0]
                Made_Y += result[1]

                # 누적 데이터량 표시
                print(file, len(Made_X))  # 현재까지 321927개
                # if len(Made_X) > 100000:
                # quit()

        # SAVING X, Y
        X = np.array(Made_X)
        Y = np.array(Made_Y)
        print(np.sum(Y))

        np.save('./Made_X/Made_X %s' % input_data_length, X)
        np.save('./Made_X/Made_Y %s' % input_data_length, Y)

        plt.plot(Y)
        plt.savefig('./Made_X/Made_Y %s.png' % input_data_length)
        plt.close()
