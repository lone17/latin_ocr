from keras.models import Sequential, Model
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization,
                          Flatten, GlobalMaxPool2D, MaxPool2D, concatenate,
                          Activation, Input, Dense, Dropout, TimeDistributed,
                          Bidirectional, LSTM, GlobalAveragePooling2D, GRU,
                          Convolution1D, MaxPool1D, GlobalMaxPool1D, MaxPooling2D,
                          Reshape, Lambda)
from keras import optimizers
from keras.utils import Sequence, to_categorical
from keras.regularizers import l2
from keras.initializers import random_normal
from keras.activations import relu
from keras import backend as K
from text_utils import *
from image_utils import *
from model_utils import *
import inspect

# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc(y_true, y_pred):
    return y_pred

def model0(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,2))(x)

    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    frame = inspect.currentframe()
    model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // down_height_factor[model_name]*128))(x)

    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.01)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def model1(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,2))(x)

    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    # x = Dropout(0.2)(x)

    frame = inspect.currentframe()
    model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // down_height_factor[model_name]*128))(x)

    x = LSTM(256, return_sequences=True, activation='tanh')(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.01)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def model2(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(8, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,2))(x)

    x = Convolution2D(16, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(32, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    frame = inspect.currentframe()
    model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // down_height_factor[model_name]*64))(x)

    x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.01)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def model3(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,2))(x)

    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

    frame = inspect.currentframe()
    model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // down_height_factor[model_name]*128))(x)

    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='sum')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    x = Dropout(0.2)(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.001)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def model4(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (10,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(32, (10,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (10,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(64, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(128, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

    frame = inspect.currentframe()
    model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // down_height_factor[model_name]*128))(x)

    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='sum')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    x = Dropout(0.2)(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.001)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def model5(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (9,19), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(16, (9,19), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(32, (9,19), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (9,19), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(64, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(128, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((1,2))(x)
    x = Dropout(0.5)(x)

    frame = inspect.currentframe()
    model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // down_height_factor[model_name]*128))(x)

    x = TimeDistributed(Dense(512, activation='elu'))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    x = Dropout(0.5)(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.001)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def model6(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (9,19), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(16, (9,19), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(32, (9,19), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (9,19), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(64, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(128, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((1,2))(x)
    x = Dropout(0.5)(x)

    frame = inspect.currentframe()
    model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // down_height_factor[model_name]*128))(x)

    x = TimeDistributed(Dense(256, activation='elu'))(x)
    # x = Dropout(0.2)(x)
    # x = LSTM(256, return_sequences=True, activation='tanh')(x)
    # x = LSTM(128, return_sequences=True, activation='tanh')(x)
    x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    # x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    x = Dropout(0.5)(x)
    # x = TimeDistributed(Dense(256, activation='relu'))(x)
    # x = Dropout(0.2)(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.001)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def model7(training=True):
    from inception_blocks_model7 import conv2d_bn, block_a, block_b
    inp = Input(shape=(None,img_h,1), name='input')

    x = block_a(inp, 8, pool_size=(2,2))
    x = block_b(x, 16, pool_size=(2,2))
    x = block_b(x, 32, pool_size=(2,2))
    x = block_b(x, 64, pool_size=(1,2))
    # x = conv2d_bn(x, 128, (5,11))
    # x = MaxPool2D((1,2))(x)

    frame = inspect.currentframe()
    model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // down_height_factor[model_name]*64*3))(x)

    x = TimeDistributed(Dense(512, activation='elu'))(x)
    # x = Dropout(0.2)(x)
    # x = LSTM(256, return_sequences=True, activation='tanh')(x)
    # x = LSTM(128, return_sequences=True, activation='tanh')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    # x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    # x = Dropout(0.5)(x)
    # x = TimeDistributed(Dense(256, activation='relu'))(x)
    # x = Dropout(0.2)(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.001)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def model_(training=True):
    from inception_blocks import conv2d_bn, block_a, block_b
    inp = Input(shape=(None,img_h,1), name='input')

    x = block_a(inp, 8, pool_size=(2,2))
    # x = block_b(x, 16, pool_size=(3,3), pool_strides=(1,1), pool_padding='same')
    x = block_b(x, 16, pool_size=(2,2))
    # x = block_b(x, 32, pool_size=(3,3), pool_strides=(1,1), pool_padding='same')
    x = block_b(x, 32, pool_size=(2,2))
    # x = block_b(x, 64, pool_size=(2,2), pool_strides=(1,1), pool_padding='same')
    x = block_b(x, 64, pool_size=(1,2))
    # x = Dropout(0.3)(x)

    # x = conv2d_bn(x, 128, (1,1), padding='same', strides=(1, 1), use_bias=False)
    frame = inspect.currentframe()
    model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // down_height_factor[model_name]*64*4))(x)

    x = TimeDistributed(Dense(512, activation='elu'))(x)
    # x = Dropout(0.2)(x)
    # x = LSTM(256, return_sequences=True, activation='tanh')(x)
    # x = LSTM(128, return_sequences=True, activation='tanh')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='sum')(x)
    # x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='sum')(x)
    # x = Dropout(0.3)(x)
    # x = TimeDistributed(Dense(256, activation='relu'))(x)
    # x = Dropout(0.2)(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.01)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

models = {
    'model0': model0,
    'model1': model1,
    'model2': model2,
    'model3': model3,
    'model4': model4,
    'model5': model5,
    'model6': model6,
    'model7': model7,
    'model_': model_,
}

model_id = 'model7'
model = models[model_id]()
print(model.summary())
