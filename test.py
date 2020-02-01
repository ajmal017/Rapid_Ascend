import pybithumb
import re
import os
import pandas as pd
from fake_useragent import UserAgent
pd.set_option('display.max_rows', 1500)

#
# Coins = pybithumb.get_tickers()
#
# for Coin in Coins:
#     ohlcv_data = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
#     print(Coin, len(ohlcv_data))

# ua = UserAgent()
# for i in range(10):
#     User_Agent = ua.random
#     print(User_Agent)

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
with open("Keys.txt") as f:
    lines = f.readlines()
    key = lines[0].strip()
    secret = lines[1].strip()
    bithumb = pybithumb.Bithumb(key, secret)
#
# import time
#
# while True:
#     try:
print(bithumb.get_balance('BTC'))
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
