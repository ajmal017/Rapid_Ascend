import pybithumb
from fake_useragent import UserAgent

# Coins = pybithumb.get_tickers()
#
# for Coin in Coins:
#     ohlcv_data = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
#     print(Coin, len(ohlcv_data))

# for i in range(10):
ua = UserAgent()
User_Agent = ua.random
print(User_Agent)
