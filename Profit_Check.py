import pandas as pd
import numpy as np
import os


def profit_check(Date, model_num) :
    temp = []
    # input_data_length = input('input data length : ')
    input_data_length = 54
    dir = './pred_ohlcv/{}_{}'.format(input_data_length, model_num)
    ohlcv_list = os.listdir(dir)

    for file in ohlcv_list:
        if file.find(Date) is not -1:  # 해당 파일이면 temp[i] 에 넣겠다.
            filename = os.path.splitext(file)
            temp.append(filename[0].split(" ")[1])

    TotalProfits = 1.0
    profit_list = []
    mean_list = []
    for Coin in temp :
        try:
            df = pd.read_excel("./BackTest/" + "%s BackTest %s.xlsx" % (Date, Coin))
            Profits = df.Profits.cumprod().iloc[-1]
            df['fluc'] = df['high'] / df['low']
            mean = df['fluc'].mean()

            if Profits > 1:
                print(Coin, Profits)

            profit_list.append(Profits)
            mean_list.append(mean)
            TotalProfits *= Profits
        except Exception as e:
            print(e)

    return TotalProfits, profit_list, mean_list


input_data_length = 54
model_num = 7
dir = './pred_ohlcv/{}_{}'.format(input_data_length, model_num)
ohlcv_list = os.listdir(dir)

Datelist = []
Date = ''
for file in ohlcv_list:
    New_Date = str(file.split()[0])
    if Date != New_Date:
        Datelist.append(New_Date)
        Date = New_Date

for Date in Datelist:
    print(Date)
    res = profit_check(Date, model_num)
    print()
