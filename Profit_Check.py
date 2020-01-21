import pandas as pd
import os


def profit_check(Date) :
    temp = []
    dir = './pred_ohlcv'
    ohlcv_list = os.listdir(dir)

    for file in ohlcv_list:
        if file.find(Date) is not -1:  # 해당 파일이면 temp[i] 에 넣겠다.
            filename = os.path.splitext(file)
            temp.append(filename[0].split(" ")[1])

    TotalProfits = 1.0
    for Coin in temp :
        try:
            df = pd.read_excel("./BackTest/" + "%s BackTest %s.xlsx" % (Date, Coin))
            Profits = df.Profits.cumprod().iloc[-1]
            if Profits != 1 :
                print(Coin, Profits)
            TotalProfits *= Profits
        except Exception as e :
            print(e)

    return TotalProfits


dir = './pred_ohlcv'
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
    profit_check(Date)
    print()
