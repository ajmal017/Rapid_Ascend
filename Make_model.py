import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot
import scipy.misc
from math import sqrt
import itertools
from IPython.display import display
from PIL import Image

input_data_length = int(input('input_data_length : '))


Made_X = np.load('./Made_X/Made_X %s.npy' % input_data_length)
Made_Y = np.load('./Made_X/Made_Y %s.npy' % input_data_length)

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

classes=np.array(("Rapid Ascending", "None"))
# reshape 할때 필요한 params
depth = 1
row = Made_X.shape[1]
col = Made_X.shape[2]

# for i in range(0, 10):
#   image = Image.fromarray(Made_X[i].astype('float32'), 'RGB')
#   image = image.resize((row * 20, col * 20))
  # image.show()

# print(emotion_labels[data.emotion[i]])

total_len = len(Made_X)
train_len = int(total_len * 0.8)
val_len = int(total_len * 0.1)
test_len = total_len - (train_len + val_len)

X_train = Made_X[:train_len].astype('float32').reshape(-1, input_data_length, col, 1)
X_val = Made_X[train_len:train_len + val_len].astype('float32').reshape(-1, input_data_length, col, 1)
X_test = Made_X[train_len + val_len:].astype('float32').reshape(-1, input_data_length, col, 1)

print(len(X_train)) # 30000 >> total * 8 / 10
print(len(X_val)) # 3500 >> 1 / 10
print(len(X_test)) # 3500  >>  rest
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


Y_train = Made_Y[:train_len].astype('float32')
Y_val = Made_Y[train_len:train_len + val_len].astype('float32')
Y_test = Made_Y[train_len + val_len:].astype('float32')
num_classes = 2
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_val = np_utils.to_categorical(Y_val, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)
print(Y_train.shape)
print(Y_val.shape)
print(Y_test.shape)

datagen = ImageDataGenerator(
    # rotation_range = 10,
    # horizontal_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest'
    )

testgen = ImageDataGenerator(
    )
datagen.fit(X_train)
batch_size = 64


# for X_batch, _ in datagen.flow(X_train, Y_train, batch_size=9):
#     for i in range(0, 9):
#         pyplot.axis('off')
#         pyplot.subplot(330 + 1 + i)
#         pyplot.imshow(X_batch[i].reshape(input_data_length, col), cmap=pyplot.get_cmap('gray'))
#     pyplot.axis('off')
#     pyplot.show()
#     break


train_flow = datagen.flow(X_train, Y_train, batch_size=batch_size)
val_flow = testgen.flow(X_val, Y_val, batch_size=batch_size)
test_flow = testgen.flow(X_test, Y_test, batch_size=batch_size)


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2
from sklearn.metrics import confusion_matrix


def FER_Model(input_shape=(input_data_length, col, 1)):
    # first input model
    visible = Input(shape=input_shape, name='input')
    num_classes = 2
    # the 1-st block
    conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), name='pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name='drop1_1')(pool1_1)

    # the 2-nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_3')(conv2_2)
    conv2_2 = BatchNormalization()(conv2_3)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), name='pool2_1')(conv2_3)
    drop2_1 = Dropout(0.3, name='drop2_1')(pool2_1)

    # the 3-rd block
    conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_1')(drop2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    pool3_1 = MaxPooling2D(pool_size=(input_data_length // 4, 1), name='pool3_1')(conv3_4)
    drop3_1 = Dropout(0.3, name='drop3_1')(pool3_1)

    # #the 4-th block
    # conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1')(drop3_1)
    # conv4_1 = BatchNormalization()(conv4_1)
    # conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2')(conv4_1)
    # conv4_2 = BatchNormalization()(conv4_2)
    # conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3')(conv4_2)
    # conv4_3 = BatchNormalization()(conv4_3)
    # conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4')(conv4_3)
    # conv4_4 = BatchNormalization()(conv4_4)
    # pool4_1 = MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
    # drop4_1 = Dropout(0.3, name = 'drop4_1')(pool4_1)

    # #the 5-th block
    # conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1')(drop4_1)
    # conv5_1 = BatchNormalization()(conv5_1)
    # conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2')(conv5_1)
    # conv5_2 = BatchNormalization()(conv5_2)
    # conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3')(conv5_2)
    # conv5_3 = BatchNormalization()(conv5_3)
    # conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4')(conv5_3)
    # conv5_3 = BatchNormalization()(conv5_3)
    # pool5_1 = MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
    # drop5_1 = Dropout(0.3, name = 'drop5_1')(pool5_1)

    # Flatten and output
    flatten = Flatten(name='flatten')(drop3_1)
    output = Dense(num_classes, activation='softmax', name='output')(flatten)

    # create model
    model = Model(inputs=visible, outputs=output)
    # summary layers
    print(model.summary())

    return model


model = FER_Model()
opt = Adam(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#       Save mobel by epoch     #
from keras.callbacks import ModelCheckpoint
filepath="./model/rapid_ascending %s.hdf5" % input_data_length
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# keras.callbacks.Callback 로 부터 log 를 받아와 history log 를 작성할 수 있다.


# we iterate 200 times over the entire training set
num_epochs = 100
history = model.fit_generator(train_flow,
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=num_epochs,
                    verbose=2,
                    callbacks=callbacks_list,
                    validation_data=val_flow,
                    validation_steps=len(X_val) / batch_size)