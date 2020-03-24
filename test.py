import pybithumb
import pandas as pd
import numpy as np
from fake_useragent import UserAgent
import time
import os
import cv2
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
import matplotlib.pyplot as plt


Made_X = np.load('Made_X/Made_X %s_%s.npy' % (30, 64), allow_pickle=True)
print(Made_X.shape)
# img = cv2.imread('./Made_Chart_all/30_66/2020-01-10 BTC_106.png')
# img2 = img.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 해리스 코너 검출 ---①
# # 값 변경해보기
# # 변화량 결과의 최대값 10% 이상의 좌표 구하기 ---②
# gray = np.float32(gray)
# # corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
# # corners = np.int0(corners)
# dst = cv2.cornerHarris(gray, 2, 3, 0.1)
# dst = cv2.dilate(dst, None)
#
# img[dst > 0.1 * dst.max()] = [0, 0, 0]
# plt.imshow(img)
# plt.show()
# #
# result = (0, 0, 0, 0)
# result1 = (1, 1, 1, 1)
# for i in range(5):
#     result = tuple(sum(elem) for elem in zip(result, result1))
#
# print(result)
# total_df = pd.DataFrame(
#     columns=['short', 'long', 'signal', 'total_profit_avg', 'plus_profit_avg', 'minus_profit_avg', 'avg_profit_avg',
#              'min_profit_avg', 'std_avg'])
#
# file_list = os.listdir('BestSet')
# print(file_list)
# txt_list = list()
# for file in file_list:
#     try:
#         if file.split('.')[1] == 'txt':
#             txt_list.append(file)
#     except:
#         pass
#
# for txt in txt_list:
#     with open("BestSet/%s" % txt) as f:
#         lines = f.readlines()
#         for index, value in enumerate(lines):
#             factors = value.split()
#             # print(factors)
#             if len(factors) == 0:
#                 continue
#             if factors[0] != txt.split()[1].split('.')[0]:
#                 continue
#             short = int(factors[0])
#             long = int(factors[1])
#             signal = int(factors[2])
#             total_profit_avg = float(factors[3])
#
#             result_df = pd.DataFrame(data=[[short, long, signal, total_profit_avg]],
#                                      columns=['short', 'long', 'signal', 'total_profit_avg'])
#             total_df = total_df.append(result_df)
#             # print(total_df)
#             # quit()
#     total_df.to_excel('./BestSet/total_df %s.xlsx' % short)
#
# file_list = os.listdir('BestSet')
# df_list = list()
# for file in file_list:
#     try:
#         if file.split('.')[1] == 'xlsx':
#             df_list.append(file)
#     except:
#         pass
#
# # print(df_list)
# # quit()
# for df in df_list:
#     result_df = pd.read_excel('./BestSet/%s' % df, index_col=0)
#     total_df = total_df.append(result_df)
#
# sorted_df = total_df.sort_values(by='std_avg', ascending=True)
# aligned_df = sorted_df.reset_index(drop=True)
# print(aligned_df.head(50).drop_duplicates())
# quit()
# aligned_df = aligned_df.head(50)
# plt.subplot(311)
# plt.scatter(aligned_df['total_profit_avg'], aligned_df['short'], )
# plt.subplot(312)
# plt.scatter(aligned_df['total_profit_avg'], aligned_df['long'], )
# plt.subplot(313)
# plt.scatter(aligned_df['total_profit_avg'], aligned_df['signal'], )
# plt.show()
    # print(lines)
    # quit()
# input_data_length = 30
# model_num = 73
# file_cnt = 1
# while True:
#     try:
#         result_x = np.load('Made_X/Made_X %s_%s %s.npy' % (input_data_length,
#                                                            model_num, file_cnt))
#         result_y = np.load('Made_X/Made_Y %s_%s %s.npy' % (input_data_length,
#                                                            model_num, file_cnt))
#         if file_cnt == 1:
#             Made_X = result_x
#             Made_Y = result_y
#         else:
#             Made_X = np.vstack((Made_X, result_x))
#             Made_Y = np.vstack((Made_Y, result_y))
#         print(Made_X.shape[0])
#
#         file_cnt += 1
#     except Exception as e:
#         print(e)
#         break
#
# print(Made_X.shape)
# print(Made_Y.shape)
# np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), (Made_X))
# np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), (Made_Y))

# arr = np.arange(10).reshape(1, 2, 5, 1)
# print(arr.shape)
# print(1 < 2 < 3)
# ar = np.arange(2)
# lis = [ar] * 3
# # print(lis)
# result = lis[0]
# for i in range(1, len(lis)):
#     result = np.vstack((result, lis[i]))
# print(result.shape)
# dir_path = './Made_Chart_to_np/30_64/'
# rename_list = os.listdir(dir_path)
# print(rename_list)
# for file in rename_list:
#     new_name = file.replace('_', ' ')
#     os.rename(dir_path + file, dir_path + new_name)


# print(('d', 'c' in list('d1')))
# for coin in pybithumb.get_ticker
# string = 'dfs.xlsx'
# print(string.endswith('.xlsx'))
# print(np.arange(10) + 1)
# test = np.arange(10)
# test[test.values]
# s():
#     df_sample = pybithumb.get_ohlcv(coin, 'KRW', 'minute1')    #.loc['2020-02-16 22:57:00', :]    # 2020-02-16 22:57:00
#     # df_sample = pd.read_excel('')
#     index = df_sample.index.values
#     print(index[-1])
# print(index[:20])
#   finding index number    #
# for i in range(len(index)):
#
#     if index[i] == np.datetime64('2020-02-15T23:19:00.000000000'):
#         print(i)
#
#         break

# start_time = time.time()
# df = pybithumb.get_ohlcv('BTC', 'KRW', 'minute1')
# print(df.index.name)
# print(1 == 1)
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
# #     print('ok')
# words_list = list(map(str.upper, ['dkj', 'df']))
# print(words_list)
# from Make_X4 import low_high
# X_test, buy_price = low_high('PCM', 54, 'proxy')
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
# TopCoin = list(series.index)[:20]
# # #
# for Coin in TopCoin:
#     ohlcv_data = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
#     if len(ohlcv_data) > 500:
#         print(Coin, len(ohlcv_data))

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
