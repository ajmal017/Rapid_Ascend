import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
import os
from Make_X2 import made_x
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)


def dividing(Y_test):
    if Y_test.shape[1] == 1:

        np_one = np.ones((len(Y_test), 1))
        np_zeros = np.zeros((len(Y_test), 1))

        if np.sum(Y_test) != 0:
            Y_test = np.concatenate((np_zeros, np_one), axis=1)

        else:
            Y_test = np.concatenate((np_one, np_zeros), axis=1)

    return Y_test


if __name__ == '__main__':

    # input_data_length = int(input("Input Data Length : "))
    input_data_length = 54
    model_num = input('Press model num : ')

    #       Make folder      #
    try:
        os.mkdir('./pred_ohlcv/%s_%s/' % (input_data_length, model_num))
        os.mkdir('./Figure_pred/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass
    except_list = os.listdir('./pred_ohlcv/%s_%s' % (input_data_length, model_num))

    #           PARAMS           #
    check_span = 30
    get_fig = 0

    #       LOAD MODEL      #
    model_low = load_model('./model/rapid_ascending_low %s_%s.hdf5' % (input_data_length, model_num))
    model_high = load_model('./model/rapid_ascending_high %s_%s.hdf5' % (input_data_length, model_num))

    for file in ohlcv_list:

        # if file in except_list:
        #     continue
        # file = '2020-01-29 FCT ohlcv.xlsx'

        print('loading %s' % file)

        try:
            X_test, Y_test_low, Y_test_high, sliced_ohlcv = made_x(file, input_data_length, model_num, check_span, get_fig)

            if len(sliced_ohlcv) < 100:
                continue

        except Exception as e:
            print('Error in getting data from made_x :', e)
            continue

        closeprice = np.roll(np.array(list(map(lambda x: x[-1][1:2][0], X_test))), -1)
        MA60 = np.roll(np.array(list(map(lambda x: x[-1][[5]][0], X_test))), -1)

        # dataX 에 담겨있는 value 에 [-1] : 바로 이전의 행 x[-1][:].shape = (1, 6)
        # sliced_ohlcv = np.array(list(map(lambda x: x[-1][:], X_test)))
        # print(sliced_ohlcv)
        # quit()

        if len(X_test) != 0:

            X_test = np.array(X_test)
            Y_test_low = np.array(Y_test_low)
            Y_test_high = np.array(Y_test_high)

            row = X_test.shape[1]
            col = X_test.shape[2]

            X_test = X_test.astype('float32').reshape(-1, row, col, 1)
            Y_test_low = np_utils.to_categorical(Y_test_low.astype('float32'))
            Y_test_high = np_utils.to_categorical(Y_test_high.astype('float32'))
            print(X_test.shape)
            print(Y_test_low.shape)
            print(Y_test_high.shape)
            # quit()

            Y_test_low = dividing(Y_test_low)
            Y_test_high = dividing(Y_test_high)

            try:
                loss = model_low.evaluate(X_test, Y_test_low)
                print("Test Loss " + str(loss[0]))
                print("Test Acc: " + str(loss[1]))
                loss = model_high.evaluate(X_test, Y_test_high)
                print("Test Loss " + str(loss[0]))
                print("Test Acc: " + str(loss[1]))

            except Exception as e:
                print('Error in model evalutate :', e)
                continue

            #       Data Preprocessing      #
            Y_pred_low_ = model_low.predict(X_test, verbose=1)
            Y_pred_high_ = model_high.predict(X_test, verbose=1)

            # max_value = np.max(Y_pred_[:, [-1]])
            max_value_low = np.max(Y_pred_low_[:, [-1]])
            max_value_high = np.max(Y_pred_high_[:, [-1]])

            # limit_line = 0.9
            limit_line_low = 0.9
            limit_line_high = 0.9

            Y_test_low = np.argmax(Y_test_low, axis=1)
            Y_test_high = np.argmax(Y_test_high, axis=1)

            Y_pred_low = np.where(Y_pred_low_[:, [-1]] > max_value_low * limit_line_low, 1, 0)
            Y_pred_high = np.where(Y_pred_high_[:, [-1]] > max_value_high * limit_line_high, 1, 0)

            #       Save Pred_ohlcv      #
            # 기존에 pybithumb 을 통해서 제공되던 ohlcv 와는 조금 다르다 >> 이전 데이터와 현재 y 데이터 행이 같다.
            sliced_Y_low = Y_pred_low.reshape(-1, 1)
            sliced_Y_high = Y_pred_high.reshape(-1, 1)
            pred_ohlcv = np.concatenate((sliced_ohlcv, sliced_Y_low, sliced_Y_high), axis=1)

            # col 이 7이 아닌 데이터 걸러주기
            try:
                pred_ohlcv_df = pd.DataFrame(pred_ohlcv, columns=['open', 'close', 'high', 'low', 'volume', 'MA60',
                                                                  'low_check', 'high_check'])
                # pred_ohlcv_df = pd.DataFrame(pred_ohlcv,
                #                              columns=['open', 'close', 'high', 'low', 'volume', 'MA60', 'fluc_close',
                #                                       'low_check', 'high_check'])

            except Exception as e:
                print('Error in making dataframe :', e)
                continue
            # print(pred_ohlcv_df.tail(20))
            # quit()
            pred_ohlcv_df.to_excel('./pred_ohlcv/%s_%s/%s' % (input_data_length, model_num, file))

            if get_fig == 1:
                spanlist_low = []
                spanlist_high = []

                for m in range(len(Y_pred_low)):
                    if Y_pred_low[m] > 0.5:
                        if m + 1 < len(Y_pred_low):
                            spanlist_low.append((m, m + 1))
                        else:
                            spanlist_low.append((m - 1, m))

                for m in range(len(Y_pred_high)):
                    if Y_pred_high[m] > 0.5:
                        print()
                        if m + 1 < len(Y_pred_high):
                            spanlist_high.append((m, m + 1))
                        else:
                            spanlist_high.append((m - 1, m))

                # ----------- 인덱스 초기화 됨 -----------#

                plt.subplot(211)
                # plt.subplot(312)
                plt.plot(closeprice, 'r', label='close')
                plt.plot(MA60, 'b', label='MA60')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_low)):
                    plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='m', alpha=0.5)

                plt.subplot(212)
                # plt.subplot(313)
                plt.plot(closeprice, 'r', label='close')
                plt.plot(MA60, 'b', label='MA60')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_high)):
                    plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='c', alpha=0.5)

                Date = file.split()[0]
                Coin = file.split()[1].split('.')[0]
                plt.savefig('./Figure_pred/%s_%s/%s %s.png' % (input_data_length, model_num, Date, Coin), dpi=500)
                plt.close()





