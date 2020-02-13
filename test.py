import pybithumb
import pandas as pd
from fake_useragent import UserAgent
import time
pd.set_option('display.max_rows', 1500)


# start_time = time.time()
# pybithumb.get_ohlcv('BTC', 'KRW', 'minute1')
# print(time.time() - start_time)
# for i in range(10):
#     start_time = time.time()
#     print(pybithumb.get_ohlcv('BTC', 'KRW', 'minute1', 'proxy'))
#     print(time.time() - start_time)
# import numpy as np
#
# nlist = np.array([2, 3, 1, 1,1, 1,1,1,1,1,1,11,1,1]).reshape(-1, 1)
# if (sum(nlist > 1) == 3):
#     if nlist[-3] > 1:
#         print('ok')
#     print(sum(nlist > 1))
#     print('ok')
from Make_X4 import low_high
X_test, buy_price = low_high('PCM', 54, 'proxy')
# nlist =
# nlist = list(filter(None, nlist))
# Fluclist = list(map(float, list(filter(None, nlist))))
# print(Fluclist)
# df = pybithumb.get_ohlcv('waves'.upper(), 'KRW', 'minute1', 'proxy')
# df.to_excel('test.xlsx')
# from Funcs_CNN4 import rsi, obv, macd
#
# print(macd(pybithumb.get_ohlcv('MTL', 'KRW', 'minute1')))
# print(obv(pybithumb.get_ohlcv('BTC', 'KRW', 'minute1')))
# from datetime import datetime
# print(datetime.now().date())

# BVC_data = pd.read_excel('BVC/2020-01-26 4.515 0.998 by 21 3 10.xlsx')
# print(BVC_data.TotalProfits.cumprod().astype('float64'))
# print(int(BVC_data.TotalProfits.cumprod().iloc[-1]))

#
# Coins = pybithumb.get_tickers()
# # #
# for Coin in Coins:
#     ohlcv_data = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
#     print(Coin, ohlcv_data.close.max() / ohlcv_data.close.min(), len(ohlcv_data))

# list = [1, 1, 1]
# print(set(list))
# ua = UserAgent()
# ua_list = []
# for i in range(10000000):
#     User_Agent = ua.random
#     ua_list.append(User_Agent)
#
# set_ua_list = set(ua_list)
# print(set_ua_list)
# print(len(set_ua_list))

from pandas.core.algorithms import value_counts

# home_dir = os.path.expanduser('~')
# dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
# ohlcv_list = os.listdir(dir)
#
# # for file in ohlcv_list:
# #     read_df = pd.read_excel(dir + file)
# #
# #     if len(read_df.columns) != 6:
# #         print(read_df.columns)
#
# df = pybithumb.get_ohlcv('FCT', 'KRW', 'minute1')
# # df.to_excel(dir + '2020-01-29 FCT ohlcv.xlsx')
# print(df)
# with open("Keys.txt") as f:
#     lines = f.readlines()
#     key = lines[0].strip()
#     secret = lines[1].strip()
#     bithumb = pybithumb.Bithumb(key, secret)
#
# import time
#
# while True:
#     try:
# print(bithumb.get_balance('BTC'))
#         num_list = re.findall("\d+", message)
#         print(int(num_list[0]) - int(num_list[1]))
#         time.sleep(1)
#
#     except:
#         print(bithumb.get_balance('BTC'))
#         break

# print('something in your', end=' ')
# print('eyes')
#
# result_df = pd.read_excel('result_df.xlsx', index_col=0)
# # result_df = result_df.sort
# result_df = result_df.sort_values(by=['profit'], ascending=False)
# result_df.to_excel('result_df.xlsx')
