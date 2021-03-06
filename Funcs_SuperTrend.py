import pybithumb
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy import stats
from asq.initiators import query
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
import time
import mpl_finance as mf
from sklearn.svm import SVR
import random

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


def transh_hour(realtime, numb):
    Hour = realtime[numb].split(':')[0]
    return int(Hour)


def transh_min(realtime, numb):
    Minute = realtime[numb].split(':')[1]
    return int(Minute)


# print(transh_min(-1))
def transh_fluc(Coin):
    try:
        TransH = pybithumb.transaction_history(Coin)
        TransH = TransH['data']
        Realtime = query(TransH).select(lambda item: item['transaction_date'].split(' ')[1]).to_list()

        # 거래 활발한지 검사
        if (transh_hour(Realtime, -1) - transh_hour(Realtime, 0)) < 0:
            if 60 + (transh_min(Realtime, -1) - transh_min(Realtime, 0)) > 30:
                return 0, 0
        elif 60 * (transh_hour(Realtime, -1) - transh_hour(Realtime, 0)) + (
                transh_min(Realtime, -1) - transh_min(Realtime, 0)) > 30:
            return 0, 0

        # 1분 동안의 거래 이력을 조사하는 루프, 0 - 59 와 같은 음수처리를 해주어야한다.
        i = 1
        while True:
            i += 1
            if i > len(Realtime):
                m = i
                break
            # 음수 처리
            if (transh_min(Realtime, -1) - transh_min(Realtime, -i)) < 0:
                if (60 + transh_min(Realtime, -1) - transh_min(Realtime, -i)) > 1:
                    m = i - 1
                    break
            elif (transh_min(Realtime, -1) - transh_min(Realtime, -i)) > 1:
                m = i - 1
                break

        # Realtime = query(TransH[-i:]).select(lambda item: item['transaction_date'].split(' ')[1]).to_list()
        Price = list(map(float, query(TransH[-m:]).select(lambda item: item['price']).to_list()))

        # print(Realtime)
        # print(Price)
        fluc = max(Price) / min(Price)
        if TransH[-1]['type'] == 'ask':
            fluc = -fluc
        return fluc, min(Price)

    except Exception as e:
        print("Error in transh_fluc :", e)
        return 0, 0


def realtime_transaction(Coin, display=5):
    Transaction_history = pybithumb.transaction_history(Coin)
    Realtime = query(Transaction_history['data'][-display:]).select(
        lambda item: item['transaction_date'].split(' ')[1]).to_list()
    Realtime_Price = list(
        map(float, query(Transaction_history['data'][-display:]).select(lambda item: item['price']).to_list()))
    Realtime_Volume = list(
        map(float, query(Transaction_history['data'][-display:]).select(lambda item: item['units_traded']).to_list()))

    print("##### 실시간 체결 #####")
    print("{:^10} {:^10} {:^20}".format('시간', '가격', '거래량'))
    for i in reversed(range(display)):
        print("%-10s %10.2f %20.3f" % (Realtime[i], Realtime_Price[i], Realtime_Volume[i]))
    return


def realtime_hogachart(Coin, display=3):
    Hogachart = pybithumb.get_orderbook(Coin)

    print("##### 실시간 호가창 #####")
    print("{:^10} {:^20}".format('가격', '거래량'))
    for i in reversed(range(display)):
        print("%10.2f %20.3f" % (Hogachart['asks'][i]['price'], Hogachart['asks'][i]['quantity']))
    print('-' * 30)
    for j in range(display):
        print("%10.2f %20.3f" % (Hogachart['bids'][j]['price'], Hogachart['bids'][j]['quantity']))


def realtime_volume(Coin):
    Transaction_history = pybithumb.transaction_history(Coin)
    Realtime_Volume = query(Transaction_history['data']).where(lambda item: item['type'] == 'bid').select(
        lambda item: item['units_traded']).to_list()
    Realtime_Volume = sum(list(map(float, Realtime_Volume)))
    return Realtime_Volume


def realtime_volume_ratio(Coin):
    Transaction_history = pybithumb.transaction_history(Coin)
    Realtime_bid = query(Transaction_history['data']).where(lambda item: item['type'] == 'bid').select(
        lambda item: item['units_traded']).to_list()
    Realtime_ask = query(Transaction_history['data']).where(lambda item: item['type'] == 'ask').select(
        lambda item: item['units_traded']).to_list()
    Realtime_bid = sum(list(map(float, Realtime_bid)))
    Realtime_ask = sum(list(map(float, Realtime_ask)))
    Realtime_Volume_Ratio = Realtime_bid / Realtime_ask
    return Realtime_Volume_Ratio


def topcoinlist(Date):
    temp = []
    dir = 'C:/Users/장재원/OneDrive/Hacking/CoinBot/ohlcv/'
    ohlcv_list = os.listdir(dir)

    for file in ohlcv_list:
        if file.find(Date) is not -1:  # 해당 파일이면 temp[i] 에 넣겠다.
            filename = os.path.splitext(file)
            temp.append(filename[0].split(" ")[1])
    return temp


def get_ma_min(Coin):
    df = pybithumb.get_ohlcv(Coin, "KRW", 'minute1')

    df['MA20'] = df['close'].rolling(20).mean()

    DatetimeIndex = df.axes[0]
    period = 20
    if inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period]) < 0:
        if 60 + (intmin(DatetimeIndex[-1]) - intmin(DatetimeIndex[-period])) > 30:
            return 0
    elif 60 * (inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period])) + intmin(DatetimeIndex[-1]) - intmin(
            DatetimeIndex[-period]) > 30:
        return 0
    slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], df.MA20[-period:])

    return slope


def get_ma20_min(Coin):
    df = pybithumb.get_ohlcv(Coin, "KRW", 'minute1')

    maxAbsScaler = MaxAbsScaler()

    df['MA20'] = df['close'].rolling(20).mean()
    MA_array = np.array(df['MA20']).reshape(len(df.MA20), 1)
    maxAbsScaler.fit(MA_array)
    scaled_MA = maxAbsScaler.transform(MA_array)

    period = 5
    slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], scaled_MA[-period:])

    return slope


def get_obv_min(Coin):
    df = pybithumb.get_ohlcv(Coin, "KRW", "minute1")

    obv = [0] * len(df.index)
    for m in range(1, len(df.index)):
        if df['close'].iloc[m] > df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1] + df['volume'].iloc[m]
        elif df['close'].iloc[m] == df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1]
        else:
            obv[m] = obv[m - 1] - df['volume'].iloc[m]
    df['OBV'] = obv

    # 24시간의 obv를 잘라서 box 높이를 만들어주어야한다.
    DatetimeIndex = df.axes[0]
    boxheight = [0] * len(df.index)
    whaleincome = [0] * len(df.index)
    for m in range(len(df.index)):
        # 24시간 시작행 찾기, obv 데이터가 없으면 stop
        n = m
        while True:
            n -= 1
            if n < 0:
                n = 0
                break
            if inthour(DatetimeIndex[m]) - inthour(DatetimeIndex[n]) < 0:
                if 60 - (intmin(DatetimeIndex[m]) - intmin(DatetimeIndex[n])) >= 60 * 24:
                    break
            elif 60 * (inthour(DatetimeIndex[m]) - inthour(DatetimeIndex[n])) + intmin(DatetimeIndex[m]) - intmin(
                    DatetimeIndex[n]) >= 60 * 24:
                break
        obv_trim = obv[n:m]
        if len(obv_trim) != 0:
            boxheight[m] = max(obv_trim) - min(obv_trim)
            if obv[m] - min(obv_trim) != 0:
                whaleincome[m] = abs(max(obv_trim) - obv[m]) / abs(obv[m] - min(obv_trim))

    df['BoxHeight'] = boxheight
    df['Whaleincome'] = whaleincome

    period = 0
    while True:
        period += 1
        if period >= len(DatetimeIndex):
            break
        if inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period]) < 0:
            if 60 + (intmin(DatetimeIndex[-1]) - intmin(DatetimeIndex[-period])) >= 10:
                break
        elif 60 * (inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period])) + intmin(DatetimeIndex[-1]) - intmin(
                DatetimeIndex[-period]) >= 10:
            break

    slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], df.OBV[-period:])
    if period < 3:
        df['Whaleincome'].iloc[-1], slope = 0, 0
    else:
        slope = slope / df['BoxHeight'].iloc[-1]

    return df['Whaleincome'].iloc[-1], slope


def GetHogaunit(Hoga):
    if Hoga < 1:
        Hogaunit = 0.0001
    elif 1 <= Hoga < 10:
        Hogaunit = 0.001
    elif 10 <= Hoga < 100:
        Hogaunit = 0.01
    elif 100 <= Hoga < 1000:
        Hogaunit = 0.1
    elif 1000 <= Hoga < 5000:
        Hogaunit = 1
    elif 5000 <= Hoga < 10000:
        Hogaunit = 5
    elif 10000 <= Hoga < 50000:
        Hogaunit = 10
    elif 50000 <= Hoga < 100000:
        Hogaunit = 50
    elif 100000 <= Hoga < 500000:
        Hogaunit = 100
    elif 500000 <= Hoga < 1000000:
        Hogaunit = 500
    else:
        Hogaunit = 1000
    return Hogaunit


def clearance(price):
    try:
        Hogaunit = GetHogaunit(price)
        Htype = type(Hogaunit)
        if Hogaunit == 0.1:
            price2 = int(price * 10) / 10.0
        elif Hogaunit == 0.01:
            price2 = int(price * 100) / 100.0
        elif Hogaunit == 0.001:
            price2 = int(price * 1000) / 1000.0
        elif Hogaunit == 0.0001:
            price2 = int(price * 10000.0) / 10000.0
        else:
            return int(price) // Hogaunit * Hogaunit
        return Htype(price2)

    except Exception as e:
        return np.nan


def inthour(date):
    date = str(date)
    date = date.split(' ')
    hour = int(date[1].split(':')[0])  # 시
    return hour


def intmin(date):
    date = str(date)
    date = date.split(' ')
    min = int(date[1].split(':')[1])  # 분
    return min


def cmo(df, period=9):
    df['closegap_cunsum'] = (df['close'] - df['close'].shift(1)).cumsum()
    df['closegap_abs_cumsum'] = abs(df['close'] - df['close'].shift(1)).cumsum()
    # print(df)

    df['CMO'] = (df['closegap_cunsum'] - df['closegap_cunsum'].shift(period)) / (
            df['closegap_abs_cumsum'] - df['closegap_abs_cumsum'].shift(period)) * 100

    del df['closegap_cunsum']
    del df['closegap_abs_cumsum']

    return df['CMO']


def rsi(ohlcv_df, period=14):
    ohlcv_df['up'] = np.where(ohlcv_df.diff(1)['close'] > 0, ohlcv_df.diff(1)['close'], 0)
    ohlcv_df['down'] = np.where(ohlcv_df.diff(1)['close'] < 0, ohlcv_df.diff(1)['close'] * (-1), 0)
    ohlcv_df['au'] = ohlcv_df['up'].rolling(period).mean()
    ohlcv_df['ad'] = ohlcv_df['down'].rolling(period).mean()
    ohlcv_df['RSI'] = ohlcv_df.au / (ohlcv_df.ad + ohlcv_df.au) * 100

    del ohlcv_df['up']
    del ohlcv_df['down']
    del ohlcv_df['au']
    del ohlcv_df['ad']

    return ohlcv_df.RSI


def obv(df):
    obv = [0] * len(df)
    for m in range(1, len(df)):
        if df['close'].iloc[m] > df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1] + df['volume'].iloc[m]
        elif df['close'].iloc[m] == df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1]
        else:
            obv[m] = obv[m - 1] - df['volume'].iloc[m]

    return obv


def macd(df, short=12, long=26, signal=9, period=3):
    df['MACD'] = df['close'].ewm(span=short, min_periods=short - 1, adjust=False).mean() - \
                 df['close'].ewm(span=long, min_periods=long - 1, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, min_periods=signal - 1, adjust=False).mean()
    df['MACD_OSC'] = df.MACD - df.MACD_Signal

    # df['OSC_dev'] = np.nan
    # for i in range(period - 1, len(df)):
    #     df['OSC_dev'].iloc[i], intercept, r_value, p_value, stderr = stats.linregress([x for x in range(period)], df.MACD_OSC[i + 1 - period:i + 1])

    df['MACD_Zero'] = 0

    return


def ema_ribbon(df, ema_1=5, ema_2=8, ema_3=13):
    df['EMA_1'] = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    df['EMA_2'] = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()
    df['EMA_3'] = df['close'].ewm(span=ema_3, min_periods=ema_3 - 1, adjust=False).mean()

    return


def ema_cross(df, ema_1=30, ema_2=60):
    df['EMA_1'] = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    df['EMA_2'] = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()

    return


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


def supertrend(df, n=100, f=3):  # df is the dataframe, n is the period, f is the factor; f=3, n=7 are commonly used.
    # Calculation of ATR
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = np.nan
    df.ix[n - 1, 'ATR'] = df['TR'][:n].mean()  # .ix is deprecated from pandas verion- 0.19
    for i in range(n, len(df)):
        df['ATR'][i] = (df['ATR'][i - 1] * (n - 1) + df['TR'][i]) / n

    # Calculation of SuperTrend
    df['Upper Basic'] = (df['high'] + df['low']) / 2 + (f * df['ATR'])
    df['Lower Basic'] = (df['high'] + df['low']) / 2 - (f * df['ATR'])
    df['Upper Band'] = df['Upper Basic']
    df['Lower Band'] = df['Lower Basic']
    for i in range(n, len(df)):
        if df['close'][i - 1] <= df['Upper Band'][i - 1]:
            df['Upper Band'][i] = min(df['Upper Basic'][i], df['Upper Band'][i - 1])
        else:
            df['Upper Band'][i] = df['Upper Basic'][i]
    for i in range(n, len(df)):
        if df['close'][i - 1] >= df['Lower Band'][i - 1]:
            df['Lower Band'][i] = max(df['Lower Basic'][i], df['Lower Band'][i - 1])
        else:
            df['Lower Band'][i] = df['Lower Basic'][i]
    df['SuperTrend'] = np.nan
    for i in df['SuperTrend']:
        if df['close'][n - 1] <= df['Upper Band'][n - 1]:
            df['SuperTrend'][n - 1] = df['Upper Band'][n - 1]
        elif df['close'][n - 1] > df['Upper Band'][i]:
            df['SuperTrend'][n - 1] = df['Lower Band'][n - 1]
    for i in range(n, len(df)):
        if df['SuperTrend'][i - 1] == df['Upper Band'][i - 1] and df['close'][i] <= df['Upper Band'][i]:
            df['SuperTrend'][i] = df['Upper Band'][i]
        elif df['SuperTrend'][i - 1] == df['Upper Band'][i - 1] and df['close'][i] >= df['Upper Band'][i]:
            df['SuperTrend'][i] = df['Lower Band'][i]
        elif df['SuperTrend'][i - 1] == df['Lower Band'][i - 1] and df['close'][i] >= df['Lower Band'][i]:
            df['SuperTrend'][i] = df['Lower Band'][i]
        elif df['SuperTrend'][i - 1] == df['Lower Band'][i - 1] and df['close'][i] <= df['Lower Band'][i]:
            df['SuperTrend'][i] = df['Upper Band'][i]

    del df['H-L']
    del df['H-PC']
    del df['L-PC']
    del df['TR']
    del df['ATR']
    del df['Upper Basic']
    del df['Lower Basic']
    del df['Upper Band']
    del df['Lower Band']

    return df


def supertrend2(df, n=100, f=3):  # df is the dataframe, n is the period, f is the factor; f=3, n=7 are commonly used.
    # Calculation of ATR
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = np.nan
    df.ix[n - 1, 'ATR'] = df['TR'][:n].mean()  # .ix is deprecated from pandas verion- 0.19
    for i in range(n, len(df)):
        df['ATR'][i] = (df['ATR'][i - 1] * (n - 1) + df['TR'][i]) / n

    # Calculation of SuperTrend
    df['Upper Basic'] = (df['high'] + df['low']) / 2 + (f * df['ATR'])
    df['Lower Basic'] = (df['high'] + df['low']) / 2 - (f * df['ATR'])
    df['Upper Band'] = df['Upper Basic']
    df['Lower Band'] = df['Lower Basic']
    for i in range(n, len(df)):
        if df['close'][i - 1] <= df['Upper Band'][i - 1]:
            df['Upper Band'][i] = min(df['Upper Basic'][i], df['Upper Band'][i - 1])
        else:
            df['Upper Band'][i] = df['Upper Basic'][i]
    for i in range(n, len(df)):
        if df['close'][i - 1] >= df['Lower Band'][i - 1]:
            df['Lower Band'][i] = max(df['Lower Basic'][i], df['Lower Band'][i - 1])
        else:
            df['Lower Band'][i] = df['Lower Basic'][i]
    df['SuperTrend2'] = np.nan
    for i in df['SuperTrend2']:
        if df['close'][n - 1] <= df['Upper Band'][n - 1]:
            df['SuperTrend2'][n - 1] = df['Upper Band'][n - 1]
        elif df['close'][n - 1] > df['Upper Band'][i]:
            df['SuperTrend2'][n - 1] = df['Lower Band'][n - 1]
    for i in range(n, len(df)):
        if df['SuperTrend2'][i - 1] == df['Upper Band'][i - 1] and df['close'][i] <= df['Upper Band'][i]:
            df['SuperTrend2'][i] = df['Upper Band'][i]
        elif df['SuperTrend2'][i - 1] == df['Upper Band'][i - 1] and df['close'][i] >= df['Upper Band'][i]:
            df['SuperTrend2'][i] = df['Lower Band'][i]
        elif df['SuperTrend2'][i - 1] == df['Lower Band'][i - 1] and df['close'][i] >= df['Lower Band'][i]:
            df['SuperTrend2'][i] = df['Lower Band'][i]
        elif df['SuperTrend2'][i - 1] == df['Lower Band'][i - 1] and df['close'][i] <= df['Lower Band'][i]:
            df['SuperTrend2'][i] = df['Upper Band'][i]

    del df['H-L']
    del df['H-PC']
    del df['L-PC']
    del df['TR']
    del df['ATR']
    del df['Upper Basic']
    del df['Lower Basic']
    del df['Upper Band']
    del df['Lower Band']

    return df


def profitage(Coin, short, long, signal, model, offset_low, offset_high, Date='2019-09-25', excel=0, get_fig=0):
    # try:
    #     df = pd.read_excel(dir + '%s %s ohlcv.xlsx' % (Date, Coin), index_col=0)
    #
    # except Exception as e:
    #     print('Error in loading ohlcv_data :', e)
    #     return 1.0, 1.0, 1.0
    if not Coin.endswith('.xlsx'):
        df = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
    else:
        df = pd.read_excel('./BestSet/Test_ohlc/%s' % Coin, index_col=0)

    macd(df, short=short, long=long, signal=signal)
    supertrend(df)
    # supertrend2(df)
    # print(df.head())
    # quit()
    df['R2G_Gap'] = np.where((df['SuperTrend'].shift(1) >= df['close'].shift(1))
                             & (df['SuperTrend'] <= df['close']), df['high'] - df['SuperTrend'], np.nan) # df['R2G_Gap'].shift(1)
    for i in range(len(df)):
        if not np.isnan(df['R2G_Gap'].iloc[i]):
            # print(df['R2G_Gap'].iloc[i])
            for j in range(i + 1, len(df)):
                if not np.isnan(df['R2G_Gap'].iloc[j]):
                    # print(df['R2G_Gap'].iloc[j])
                    for k in range(i, j):
                        df['R2G_Gap'].iloc[k] = df['R2G_Gap'].iloc[i]

                    break

                elif j == len(df) - 1:
                    for k in range(i, j + 1):
                        df['R2G_Gap'].iloc[k] = df['R2G_Gap'].iloc[i]

                    break
    # print(df['R2G_Gap'])
    # quit()

    none_indexs = sum(np.isnan(df['MACD_OSC'].values))

    #               svm regression             #
    # poly_reg = PolynomialFeatures(degree=8)
    # X_poly = poly_reg.fit_transform(np.arange(len(df['MACD_OSC']) - none_indexs).reshape(-1, 1))
    x = np.arange(len(df['MACD_OSC']))[none_indexs:].reshape(-1, 1)
    print(x.shape)
    # quit()

    y = df['MACD_OSC'].values[none_indexs:].reshape(-1, 1)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y)
    print(y.shape)
    # print(1e3)
    # quit()
    model.fit(x, y)
    osc_reg = model.predict(x)
    # print(osc_reg)
    # osc_reg_inv = scaler.inverse_transform(osc_reg.reshape(-1, 1))
    osc_reg = [np.nan] * none_indexs + list(osc_reg)
    # print()
    # quit()

    #       close reg       #
    # x = np.arange(len(df['close'])).reshape(-1, 1)
    # print(x.shape)
    # quit()
    # y = df['close'].values.reshape(-1, 1)
    #
    # scaler_x = StandardScaler()
    # scaler_y = StandardScaler()
    # x = scaler_x.fit_transform(x)
    # y = scaler_y.fit_transform(y)
    # print(y.shape)
    # # print(1e3)
    # # quit()
    # model.fit(x, y)
    # close_reg = model.predict(x)
    # close_reg = scaler_y.inverse_transform(close_reg.reshape(-1, 1))
    # print(osc_reg)

    # osc_reg_inv = scaler.inverse_transform(osc_reg.reshape(-1, 1))

    plt.subplot(311)
    plt.plot(df['close'].values)
    # print(df['close'])

    span_list = list()
    span_list2 = list()
    for i in range(1, len(df)):
        # if close_reg[i - 1] <= close_reg[i]:
        #     span_list.append((i, i + 1))
        if osc_reg[i - 1] <= osc_reg[i]:
            span_list2.append((i, i + 1))
    #
    # for i in range(len(span_list)):
    #     plt.axvspan(span_list[i][0], span_list[i][1], facecolor='g', alpha=0.7)
    for i in range(len(span_list2)):
        plt.axvspan(span_list2[i][0], span_list2[i][1], facecolor='c', alpha=0.7)

    plt.subplot(312)
    # plt.plot(close_reg)
    plt.plot(df['MACD_OSC'].values, 'g')
    plt.plot(df['MACD_Zero'].values)

    # span_list = list()
    # span_list2 = list()
    # for i in range(len(df)):
    #     if df['OSC_dev'].iloc[i] < 0:
    #         span_list.append((i, i + 1))
    #     if df['MACD_OSC'].iloc[i] > 0.:
    #         span_list2.append((i, i + 1))
    #
    # for i in range(len(span_list)):
    #     plt.axvspan(span_list[i][0], span_list[i][1], facecolor='g', alpha=0.7)
    # for i in range(len(span_list2)):
    #     plt.axvspan(span_list2[i][0], span_list2[i][1], facecolor='m', alpha=0.7)

    plt.subplot(313)
    plt.plot(osc_reg)
    plt.plot(df['MACD_Zero'].values)

    for i in range(len(span_list2)):
        plt.axvspan(span_list2[i][0], span_list2[i][1], facecolor='c', alpha=0.7)
    # for i in range(len(span_list2)):
    #     plt.axvspan(span_list2[i][0], span_list2[i][1], facecolor='m', alpha=0.7)

    plt.show()
    # quit()

    # 매수 시점 = 급등 예상시, 매수가 = 이전 종가
    df['BuyPrice'] = np.where((df['MACD_OSC'] > 0.)
                              # & (df['SuperTrend'].shift(1) > df['close'].shift(1))
                              & (df['SuperTrend'] < df['close']), df['SuperTrend'] + df['R2G_Gap'] * offset_low, np.nan)
    # df['BuyPrice'] = df['BuyPrice'].apply(clearance)

    # 거래 수수료
    fee = 0.005

    # DatetimeIndex 를 지정해주기 번거로운 상황이기 때문에 틱을 기준으로 거래한다.
    # DatetimeIndex = df.axes[0]

    # ------------------- 상향 / 하향 매도 여부와 이익률 계산 -------------------#

    # high 가 SPP 를 건드리거나, low 가 SPM 을 건드리면 매도 체결 [ 매도 체결될 때까지 SPP 와 SPM 은 유지 !! ]
    length = len(df.index) - 1  # 데이터 갯수 = 1, m = 0  >> 데이터 갯수가 100 개면 m 번호는 99 까지 ( 1 - 100 )

    # 병합할 dataframe 초기화
    bprelay = pd.DataFrame(index=df.index, columns=['bprelay'])
    condition = pd.DataFrame(index=df.index, columns=["Condition"])
    Profits = pd.DataFrame(index=df.index, columns=["Profits"])
    # price_point = pd.DataFrame(index=np.arange(len(df)), columns=['Price_point'])

    Profits.Profits = 1.0
    Minus_Profits = 1.0

    # 오더라인과 매수가가 정해진 곳에서부터 일정시간까지 오더라인과 매수가를 만족할 때까지 대기  >> 일정시간,
    m = 0
    while m <= length:

        while True:  # bp 찾기
            if pd.notnull(df.iloc[m]['BuyPrice']):
                break
            m += 1
            if m > length:
                break

        if (m > length) or pd.isnull(df.iloc[m]['BuyPrice']):
            break
        bp = df.iloc[m]['BuyPrice']

        #       매수 등록 완료, 매수 체결 대기     #

        start_m = m
        while True:
            bprelay["bprelay"].iloc[m] = bp

            # 매수 체결 조건
            if df.iloc[m]['low'] <= bp:  # and (df.iloc[m]['high'] != df.iloc[m]['low']):  # 조건을 만족할 경우 spp 생성
                # print(df['low'].iloc[m], '<=', bp, '?')

                condition.iloc[m] = '매수 체결'
                # bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

                m += 1
                break
            else:
                m += 1
                if m > length:
                    break

                #       매수 대기 term      #
                if df['SuperTrend'].iloc[m] > df['close'].iloc[m]:
                # if m - start_m > 3:
                    break

                condition.iloc[m] = '매수 대기'
                bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

    # df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    # df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)

    # if excel == 1:
    #     df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
    #     quit()

    #           지정 매도가 표시 완료           #

    #                               수익성 검사                                  #

    m = 0
    spanlist = []
    spanlist_limit = []
    spanlist_breakaway = []
    while m <= length:  # 초반 시작포인트 찾기

        while True:  # SPP 와 SPM 찾긴
            if condition.iloc[m]['Condition'] == '매수 체결':
                # and type(df.iloc[m]['SPP']) != str:  # null 이 아니라는 건 오더라인과 매수가로 캡쳐했다는 거
                break
            m += 1
            if m > length:  # 차트가 끊나는 경우, 만족하는 spp, spm 이 없는 경우
                break

        if (m > length) or pd.isnull(condition.iloc[m]['Condition']):
            break

        start_m = m

        #           매수 체결 지점 확인          #

        #       Predict value == 1 지점으로 부터 일정 시간 (10분) 지난 후 checking      #
        # while m - start_m <= over_tick:
        #     m += 1
        #     if m > length:
        #         break
        #     condition.iloc[m] = '매도 대기'
        #     bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]
        # if m > length:
        #     break

        while True:

            #       고점 예측시 매도       #
            # if df['high'].iloc[m] >= df['SuperTrend'].iloc[m - 1] + df['R2G_Gap'].iloc[m - 1] * offset_high:
            if df['high'].iloc[m] >= bprelay['bprelay'].iloc[m - 1] * 1.015:
                break

            #         SuperTrend2 Green2Red     #
            elif df['SuperTrend'].iloc[m - 1] > df['close'].iloc[m - 1]:
                break

            m += 1
            if m > length:
                break

            condition.iloc[m] = '매도 대기'
            bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

        if m > length:
            # print(condition.iloc[m - 1])
            break

        # elif (df['high'].iloc[m] >= df['SuperTrend'].iloc[m - 1] + df['R2G_Gap'].iloc[m - 1] * offset_high) or (df['SuperTrend'].iloc[m - 1] > df['close'].iloc[m - 1]):
        elif (df['high'].iloc[m] >= bprelay['bprelay'].iloc[m - 1] * 1.015) or (df['SuperTrend'].iloc[m - 1] > df['close'].iloc[m - 1]):

            condition.iloc[m] = "매도 체결"

            if df['high'].iloc[m] >= df['SuperTrend'].iloc[m - 1] + df['R2G_Gap'].iloc[m - 1] * offset_high:
                Profits.iloc[m] = (df['SuperTrend'].iloc[m - 1] + df['R2G_Gap'].iloc[m - 1] * offset_high) / bprelay["bprelay"].iloc[m] - fee

            else:
                Profits.iloc[m] = df.iloc[m]['open'] / bprelay["bprelay"].iloc[m] - fee

            if float(Profits.iloc[m]) < 1:
                Minus_Profits *= float(Profits.iloc[m])
                try:
                    if start_m == m:
                        m += 1
                    spanlist.append((start_m, m))
                    spanlist_breakaway.append((start_m, m))

                except Exception as e:
                    pass

            else:
                try:
                    if start_m == m:
                        m += 1
                    spanlist.append((start_m, m))
                    spanlist_limit.append((start_m, m))

                except Exception as e:
                    pass

        # 체결시 재시작
        m += 1

    df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)
    # df = pd.merge(df, price_point, how='outer', left_index=True, right_index=True)
    df = df.reset_index(drop=True)

    if excel == 1:
        # df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
        df.to_excel("ExcelCheck/%s BackTest %s.xlsx" % (Date, Coin))

    profits = Profits.cumprod()  # 해당 열까지의 누적 곱!
    # print(profits)

    if np.isnan(profits.iloc[-1].item()):
        return 1.0, 1.0, 1.0, 1.0, 1.0

    # [-1] 을 사용하려면 데이터가 존재해야 되는데 데이터 전체가 결측치인 경우가 존재한다.
    if len(profits) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0

    elif float(profits.iloc[-1]) != 1.0 and get_fig == 1:

        # 거래 체결마다 subplot 1,2 저장
        fig = plt.figure(figsize=(10, 7))

        ax = fig.add_subplot(211)

        ochl = df.iloc[:, :4]
        index = np.arange(len(ochl))
        ochl = np.hstack((np.reshape(index, (-1, 1)), ochl))
        mf.candlestick_ochl(ax, ochl, width=0.5, colorup='r', colordown='b')
        plt.plot(df['SuperTrend'], 'blue', label='ST', linewidth=1.0)
        plt.plot(df['SuperTrend'] + df['R2G_Gap'] * offset_low , 'c', label='ST low', linewidth=1.0)
        plt.plot(df['SuperTrend'] + df['R2G_Gap'] * offset_high, 'm', label='ST high', linewidth=1.0)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='upper right', fontsize=10)
        plt.title('%.2f %.2f %.2f' % (float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits))

        for trade_num in range(len(spanlist_limit)):
            plt.axvspan(spanlist_limit[trade_num][0], spanlist_limit[trade_num][1], facecolor='c', alpha=0.5)

        for trade_num in range(len(spanlist_breakaway)):
            plt.axvspan(spanlist_breakaway[trade_num][0], spanlist_breakaway[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(212)

        plt.plot(df[['MACD_OSC']], 'gold', label='oscillator', linewidth=1.0)
        plt.plot(df[['MACD_Zero']], 'g', label='zero', linewidth=1.0)

        # plot 저장 & 닫기
        plt.show()
        # plt.savefig("./Figure_pred/%s_%s_%s/%s %s.png" % (short, long, signal, Date, Coin), dpi=300)
        # plt.close()

    if profits.values[-1] != 1.:
        profits_sum = 0
        for i in range(len(Profits)):
            if Profits.values[i] != 1.:
                profits_sum += Profits.iloc[i]
        profits_avg = profits_sum / sum(Profits.values != 1.)

    else:
        profits_avg = [1.]

    # print(Profits.values.min())
    # quit()
    # print(Profits.values != 1.)
    # print(profits_avg)
    # quit()
    # print(std)

    return float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits, profits_avg[
        0], Profits.values.min()


if __name__ == "__main__":

    home_dir = os.path.expanduser('~')
    dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
    ohlcv_list = os.listdir(dir)

    # Coinlist = pybithumb.get_tickers()
    # Fluclist = []
    # while True:
    #     try:
    #         for Coin in Coinlist:
    #             tickerinfo = pybithumb.PublicApi.ticker(Coin)
    #             data = tickerinfo['data']
    #             fluctate = data['fluctate_rate_24H']
    #             Fluclist.append(fluctate)
    #             time.sleep(1 / 90)
    #         break
    #
    #     except Exception as e:
    #         Fluclist.append(None)
    #         print('Error in making Topcoin :', e)
    #
    # Fluclist = list(map(float, Fluclist))
    # series = pd.Series(Fluclist, Coinlist)
    # series = series.sort_values(ascending=False)
    #
    # series = series[:10]
    # TopCoin = list(series.index)
    TopCoin = ['ORBS', 'COSM', 'STRAT', 'ELF', 'ENJ']
    # TopCoin = ['ORBS']

    #
    # print(Date)
    # quit()
    #     # Date = '2020-02-27'

    # ohlcv_list = ['2019-10-05 LAMB ohlcv.xlsx']

    # for file in ohlcv_list:
    #     Coin = file.split()[1].split('.')[0]
    #     Date = file.split()[0]
    Date = str(datetime.now()).split()[0]
    excel_list = os.listdir('./BestSet/Test_ohlc/')
    # random.shuffle(excel_list)
    # print(excel_list)
    # quit()
    total_df = pd.DataFrame(
        columns=['short', 'long', 'signal', 'total_profit_avg', 'plus_profit_avg', 'minus_profit_avg', 'avg_profit_avg',
                 'min_profit_avg'])

    model = SVR(kernel='rbf', C=1000, gamma=10, epsilon=0.1)

    for short in range(20, 100, 5):
        for long in range(short + 3, short + 100, 5):
            for signal in range(5, 50, 3):

                #       Make folder      #
                # try:
                #     os.mkdir("./Figure_pred/%s_%s_%s/" % (short, long, signal))
                #
                # except Exception as e:
                #     print(e)
                # short, long, signal = 12, 26, 9
                short, long, signal = 105, 168, 32

                #
                for Coin in excel_list:
                    # print(Coin)
                    # Coin = excel_list[3]
                    print(Coin, profitage(Coin, short, long, signal, model, offset_low=0.2, offset_high=1.1, Date=Date, get_fig=0, excel=0))

                total_profit = 0
                plus_profit = 0
                minus_profit = 0
                avg_profit = 0
                min_profit = 0
                for Coin in excel_list:
                    start = time.time()
                    result = profitage(Coin, short, long, signal, offset_low=1.01, offset_high=1.06, Date=Date, get_fig=1)
                    # quit()
                    total_profit += result[0]
                    plus_profit += result[1]
                    minus_profit += result[2]
                    avg_profit += result[3]
                    min_profit += result[4]
                total_profit_avg = total_profit / len(excel_list)
                plus_profit_avg = plus_profit / len(excel_list)
                minus_profit_avg = minus_profit / len(excel_list)
                avg_profit_avg = avg_profit / len(excel_list)
                min_profit_avg = min_profit / len(excel_list)
                print(short, long, signal, total_profit_avg, plus_profit_avg, minus_profit_avg, avg_profit_avg,
                      min_profit_avg, '%.3f second' % (time.time() - start))

                result_df = pd.DataFrame(data=[
                    [short, long, signal, total_profit_avg, plus_profit_avg, minus_profit_avg, avg_profit_avg,
                     min_profit_avg]],
                                         columns=['short', 'long', 'signal', 'total_profit_avg', 'plus_profit_avg',
                                                  'minus_profit_avg', 'avg_profit_avg', 'min_profit_avg'])
                total_df = total_df.append(result_df)

            total_df.to_excel('./BestSet/total_df %s.xlsx' % short)
            break
