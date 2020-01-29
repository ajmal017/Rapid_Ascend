import pybithumb
import Funcs_CNN2
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
from keras.models import load_model
import os
from Make_X2 import low_high
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#       KEY SETTING     #
with open("Keys.txt") as f:
    lines = f.readlines()
    key = lines[0].strip()
    secret = lines[1].strip()
    bithumb = pybithumb.Bithumb(key, secret)

#       Params      #
input_data_length = 54
limit_line_low = 0.9
limit_line_high = 0.9
model_num = 3

#       Trade Info      #
#   Check money    #
buy_wait = 10
Profits = 1.0

#       Model Fitting       #
model_low = load_model('./model/rapid_ascending_low %s_%s.hdf5' % (input_data_length, model_num))
model_high = load_model('./model/rapid_ascending_high %s_%s.hdf5' % (input_data_length, model_num))

while True:

    while True:
        #   이전 ohlcv 데이터로 low_check prediction 결과값이 1 이면, 매수 등록 진행
        for Coin in pybithumb.get_tickers():

            try:
                while True:
                    if datetime.now().second >= 5:
                        break

                #   proxy 설정으로 크롤링 우회한다.
                print('Loading %s low_high' % Coin)
                time.sleep(random.random() * 5)
                X_test, buy_price = low_high(Coin, input_data_length)

                #   ohlcv_data_length 가 100 이하이면 predict 하지 않는다.
                if X_test is not None:
                    Y_pred_low_ = model_low.predict(X_test, verbose=1)
                    max_value_low = np.max(Y_pred_low_[:, [-1]])
                    Y_pred_low = np.where(Y_pred_low_[:, [-1]] > max_value_low * limit_line_low, 1, 0)

                    if Y_pred_low[-1] > 0.5:
                        break

            except Exception as e:
                print('Error in %s low predict :' % Coin, e)
                continue

        if Y_pred_low[-1] > 0.5:
            break

    #                매수 등록                  #
    try:
        # 호가단위, 매수가, 수량
        limit_buy_price = Funcs_CNN2.clearance(buy_price)

        # -------------- 보유 원화 확인 --------------#
        balance = bithumb.get_balance(Coin)
        krw = balance[2]
        print("보유 원화 : ", krw)
        # money = krw * 0.996
        money = 3000
        print("주문 원화 : ", money)
        if krw < 1000:
            print("거래가능 원화가 부족합니다.\n")
            continue
        print()

        # 매수량
        buyunit = int((money / limit_buy_price) * 10000) / 10000.0

        # 매수 등록
        BuyOrder = bithumb.buy_limit_order(Coin, limit_buy_price, buyunit, "KRW")
        print("      %s %s KRW 매수 등록      " % (Coin, limit_buy_price))
        print(BuyOrder)

    except Exception as e:
        print("매수 등록 중 에러 발생 :", e)
        continue

    #                   매수 대기                    #

    start = time.time()
    Complete = 0
    while True:
        try:
            #       BuyOrder is dict / in Error 걸러주기        #
            if type(BuyOrder) != tuple:
                break

            balance = bithumb.get_balance(Coin)

            # 반 이상 체결된 경우
            if balance[0] / buyunit >= 0.5:
                Complete = 1
                # 부분 체결되지 않은 미체결 잔량을 주문 취소
                print("    매수 체결    ")
                CancelOrder = bithumb.cancel_order(BuyOrder)
                print("부분 매수 체결 : ", CancelOrder)
                print()
                time.sleep(1 / 80)
                break

            #   최대 10분 동안 매수 체결을 대기한다.
            if time.time() - start > buy_wait * 60:

                if balance[0] * limit_buy_price > 1000:
                    Complete = 1
                    print("    매수 체결    ")
                    CancelOrder = bithumb.cancel_order(BuyOrder)
                else:
                    if bithumb.get_outstanding_order(BuyOrder) is None:
                        if type(BuyOrder) == tuple:
                            # 한번 더 검사 (get_outstanding_order 찍으면서 체결되는 경우가 존재한다.)
                            if balance[0] * limit_buy_price > 1000:
                                Complete = 1
                                print("    매수 체결    ")
                                CancelOrder = bithumb.cancel_order(BuyOrder)
                            else:
                                print("매수가 취소되었습니다.\n")
                        else:
                            # BuyOrder is None 에도 체결된 경우가 존재함
                            # 1000 원은 거래 가능 최소 금액
                            if balance[0] * limit_buy_price > 1000:
                                Complete = 1
                                print("    매수 체결    ")
                                CancelOrder = bithumb.cancel_order(BuyOrder)
                            else:
                                print("미체결 또는 체결량 1000 KRW 이하\n")
                    else:
                        if type(BuyOrder) == tuple:
                            CancelOrder = bithumb.cancel_order(BuyOrder)
                            print("미체결 또는 체결량 1000 KRW 이하")
                            print(CancelOrder)
                            print()
                break

        except Exception as e:
            print('매수 체결 여부 확인중 에러 발생 :', e)

    # 지정 시간 초과로 루프 나온 경우
    if Complete == 0:
        print()
        continue

    #                    매수 체결                     #
    else:

        #           매도 대기           #
        while True:
            # 분당 한번 high_check predict 했을때, 1 결과값이 출력되면 매도 진행
            try:
                #   이전 종가가 dataframe 으로 완성될 시기
                if datetime.now().second == 55:
                    while True:
                        try:
                            X_test, _ = low_high(Coin, input_data_length)
                            break

                        except Exception as e:
                            print('Error in getting low_high data :', e)
                            time.sleep(random.random() * 5)

                    Y_pred_high_ = model_high.predict(X_test, verbose=3)
                    max_value_high = np.max(Y_pred_high_[:, [-1]])
                    Y_pred_high = np.where(Y_pred_high_[:, [-1]] > max_value_high * limit_line_high, 1, 0)

                    #   매도 진행
                    if Y_pred_high[-1] > 0.5:
                        balance = bithumb.get_balance(Coin)
                        sellunit = int((balance[0]) * 10000) / 10000.0
                        SellOrder = bithumb.sell_market_order(Coin, sellunit, 'KRW')
                        print("    %s 시장가 매도     " % Coin)
                        # SellOrder = bithumb.sell_limit_order(Coin, limit_sell_pricePlus, sellunit, "KRW")
                        # print("##### %s %s KRW 지정 매도 재등록 #####" % (Coin, limit_sell_pricePlus))
                        print(SellOrder)
                        break

            except Exception as e:
                print('Error in %s high predict :' % Coin, e)

        sell_switch = 0
        while True:

            #                   매도 재등록                    #
            try:
                if sell_switch == 1:

                    #   SellOrder Initializing
                    CancelOrder = bithumb.cancel_order(SellOrder)
                    if CancelOrder is False:  # 남아있는 매도 주문이 없다. 취소되었거나 체결완료.
                        print("    매도 체결    ")
                        Profits *= bithumb.get_balance(Coin)[2] / krw
                        print("Accumulated Profits : %.6f\n" % Profits)
                        break
                    elif CancelOrder is None:  # SellOrder = none 인 경우
                        # 매도 재등록 해야함 ( 등록될 때까지 )
                        pass
                    else:
                        print("    매도 취소    ")
                        print(CancelOrder)
                    print()

                    balance = bithumb.get_balance(Coin)
                    sellunit = int((balance[0]) * 10000) / 10000.0
                    SellOrder = bithumb.sell_market_order(Coin, sellunit, 'KRW')
                    print("    %s 시장가 매도     " % Coin)
                    # SellOrder = bithumb.sell_limit_order(Coin, limit_sell_pricePlus, sellunit, "KRW")
                    # print("##### %s %s KRW 지정 매도 재등록 #####" % (Coin, limit_sell_pricePlus))
                    print(SellOrder)

                    sell_switch = -1
                    time.sleep(1 / 80)  # 너무 빠른 거래로 인한 오류 방지

                    # 지정 매도 에러 처리
                    if type(SellOrder) in [tuple, str]:
                        pass
                    elif SellOrder is None:  # 매도 함수 자체 오류 ( 서버 문제 같은 경우 )
                        sell_switch = 1
                        continue
                    else:  # dictionary
                        # 체결 여부 확인 로직
                        ordersucceed = bithumb.get_balance(Coin)
                        if ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0:
                            print("    매도 체결    ")
                            Profits *= bithumb.get_balance(Coin)[2] / krw
                            print("Accumulated Profits : %.6f\n" % Profits)
                            break
                        else:
                            sell_switch = 1
                            continue

            except Exception as e:
                print("매도 재등록 중 에러 발생 :", e)
                sell_switch = 1
                continue

            #       매도 상태 Check >> 취소 / 체결완료 / SellOrder = None / dictionary      #
            try:
                if bithumb.get_outstanding_order(SellOrder) is None:
                    # 서버 에러에 의해 None 값이 발생할 수 도 있음..
                    if type(SellOrder) in [tuple, str]:  # 서버에러는 except 로 가요..
                        try:
                            # 체결 여부 확인 로직
                            ordersucceed = bithumb.get_balance(Coin)
                            if ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0:
                                print("    매도 체결    ")
                                Profits *= bithumb.get_balance(Coin)[2] / krw
                                print("Accumulated Profits : %.6f\n" % Profits)
                                break
                            elif bithumb.get_outstanding_order(SellOrder) is not None:  # 혹시 모르는 미체결
                                continue
                            else:
                                print("매도 주문이 이미 취소되었습니다.\n")
                                if sell_switch in [0, -1]:
                                    sell_switch = 1
                                # elif sell_switch == 0:
                                #     sppswitch = 1
                                    time.sleep(random.random() * 5)
                                continue  # 지정 매도 재등록 완료하면 다시 현재가 비교로

                        except Exception as e:
                            print('SellOrder == tuple, str? 에서 에러발생 :', e)
                            time.sleep(random.random() * 5)  # 서버 에러인 경우
                            continue

                    # 매도 등록 에러라면 ? 제대로 등록 될때까지 재등록 ! 지정 에러, 하향 에러
                    elif SellOrder is None:  # limit_sell_order 가 아예 안되는 경우
                        if sell_switch in [0, -1]:
                            sell_switch = 1
                        # elif sell_switch == 0:
                        #     sppswitch = 1
                            time.sleep(random.random() * 5)
                        continue

                    else:  # dictionary
                        # 체결 여부 확인 로직
                        ordersucceed = bithumb.get_balance(Coin)
                        if ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0:
                            print("    매도 체결    ")
                            Profits *= bithumb.get_balance(Coin)[2] / krw
                            print("Accumulated Profits : %.6f\n" % Profits)
                            break
                        else:
                            if sell_switch in [0, -1]:
                                sell_switch = 1
                            # elif sell_switch == 0:
                            #     sppswitch = 1
                                time.sleep(random.random() * 5)
                            continue

                else:  # 미체결량 대기 파트
                    if sell_switch == 0:  # 이탈 매도 대기
                        sell_switch = 1
                        time.sleep(random.random() * 5)
                    # else:  # 지정 매도 대기
                    #     time.sleep(1 / 80)

            except Exception as e:  # 지정 매도 대기 중 에러나서 이탈 매도가로 팔아치우는거 방지하기 위함.
                print('취소 / 체결완료 / SellOrder = None / dict 확인중 에러 발생 :', e)
                continue

