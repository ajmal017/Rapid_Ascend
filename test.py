import pybithumb
import os
import pandas as pd
from fake_useragent import UserAgent

# Coins = pybithumb.get_tickers()
#
# for Coin in Coins:
#     ohlcv_data = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
#     print(Coin, len(ohlcv_data))
# ua = UserAgent()
#
# for i in range(10):
#     User_Agent = ua.random
#     print(User_Agent)

from pandas.core.algorithms import value_counts
s = pd.Series([1, 1, 2, 3, 1, 2, 3, 2, 1])

print(s.apply(lambda x: x[-5:].value_counts()))



