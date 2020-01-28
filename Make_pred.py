import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
import os
from Make_X import made_x
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)

#           PARAMS           #
Range_fluc = 1.035
get_fig = 0

if __name__ == '__main__':

    input_data_length = int(input("Input Data Length : "))
    except_list = os.listdir('./pred_ohlcv/%s' % input_data_length)

    #       LOAD MODEL      #
    model = load_model('./model/rapid_ascending %s.hdf5' % input_data_length)

    for file in ohlcv_list:

        if file in except_list:
            continue

        print('loading %s' % file)

        try:
            X_test, Y_test, sliced_ohlcv = made_x(file, input_data_length, Range_fluc, get_fig)

        except Exception as e:
            print(e)
            continue

        closeprice = np.roll(np.array(list(map(lambda x: x[-1][1:2][0], X_test))), -1)
        MA60 = np.roll(np.array(list(map(lambda x: x[-1][[5]][0], X_test))), -1)

        # dataX 에 담겨있는 value 에 [-1] : 바로 이전의 행 x[-1][:].shape = (1, 6)
        # sliced_ohlcv = np.array(list(map(lambda x: x[-1][:], X_test)))
        # print(sliced_ohlcv)
        # quit()

        if len(X_test) != 0:

            X_test = np.array(X_test)
            Y_test = np.array(Y_test)

            row = X_test.shape[1]
            col = X_test.shape[2]

            X_test = X_test.astype('float32').reshape(-1, row, col, 1)
            Y_test = np_utils.to_categorical(Y_test.astype('float32'))
            print(X_test.shape)
            print(Y_test.shape)

            if Y_test.shape[1] == 1:

                np_one = np.ones((len(Y_test), 1))
                np_zeros = np.zeros((len(Y_test), 1))

                if np.sum(Y_test) != 0:
                    Y_test = np.concatenate((np_zeros, np_one), axis=1)

                else:
                    Y_test = np.concatenate((np_one, np_zeros), axis=1)

            try:
                loss = model.evaluate(X_test, Y_test)
                print("Test Loss " + str(loss[0]))
                print("Test Acc: " + str(loss[1]))

            except Exception as e:
                print(e)
                continue

            Y_pred_ = model.predict(X_test, verbose=1)
            Y_test = np.argmax(Y_test, axis=1)
            Y_pred = np.argmax(Y_pred_, axis=1)

            #       Save Pred_ohlcv      #
            # 기존에 pybithumb 을 통해서 제공되던 ohlcv 와는 조금 다르다 >> 이전 데이터와 현재 y 데이터 행이 같다.
            sliced_Y = Y_pred.reshape(-1, 1)
            pred_ohlcv = np.concatenate((sliced_ohlcv, sliced_Y), axis=1)  # axis=1 가로로 합친다

            # col 이 7이 아닌 데이터 걸러주기
            try:
                pred_ohlcv_df = pd.DataFrame(pred_ohlcv, columns=['open', 'close', 'high', 'low', 'volume', 'MA60', 'fluc_close'])

            except Exception as e:
                print(e)
                continue
            # print(pred_ohlcv_df.tail(20))
            # quit()
            pred_ohlcv_df.to_excel('./pred_ohlcv/%s/%s' % (input_data_length, file))

            # Y_pred 에 1 이 존재하면, Plot Comparing
            if np.sum(Y_pred) != 0:

                plt.figure(figsize=(10, 10))
                plt.subplot(211)
                plt.plot(Y_test, 'purple', label='test')
                plt.plot(MA60, 'b')
                plt.plot(closeprice, 'y')

                plt.subplot(212)
                plt.plot(Y_pred, 'purple', label='pred')
                plt.plot(MA60, 'b')
                plt.plot(closeprice, 'y')
                # plt.show()
                plt.savefig('./Figure_fluc/%s/%s.png' % (input_data_length, file.split('.')[0]))
                plt.close()

            print()





