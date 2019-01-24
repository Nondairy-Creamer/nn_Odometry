from tensorflow.keras.layers import multiply, Dropout, concatenate, LSTM, Dense, Input, Activation, BatchNormalization, Conv3D, subtract, add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import *
import h5py


def load_data_rr(path):
    data_in = h5py.File(path, 'r')

    train_set = data_in['train_set'][:]
    dev_set = data_in['dev_set'][:]
    test_set = data_in['test_set'][:]

    train_ans = data_in['train_ans'][:]
    dev_ans = data_in['dev_ans'][:]
    test_ans = data_in['test_ans'][:]

    sample_freq = data_in['sample_freq'][:]
    phase_step = data_in['phase_step'][:]

    data_in.close()

    return train_set, dev_set, test_set, train_ans, dev_ans, test_ans, sample_freq, phase_step


def r2(y_true, y_pred):
    r2_value = 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))
    return r2_value


def ln_model(input_shape, filter_shape, num_filter=(8, 6)):
    learning_rate = 0.001*1
    batch_size = np.power(2, 6)

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    pad_x = int((filter_shape[2] - 1) / 2)
    pad_y = int((filter_shape[1] - 1) / 2)
    pad_t = int((filter_shape[0] - 1) / 2)

    d_rate = 0.0
    reg_val = 0.001

    # T4/T5
    # intial convolution
    conv1 = Conv3D(num_filter[0], filter_shape, strides=(1, 1, 1), name='T4_T5',
                   kernel_initializer=glorot_uniform(seed=None),
                   kernel_regularizer=regularizers.l1(reg_val))(image_in)
    #conv1 = BatchNormalization()(conv1)
    conv1_a = Activation('relu')(conv1)
    conv1_d = Dropout(d_rate)(conv1_a)

    # LPTCs
    # convolution that takes up all of space
    conv_x_size = int(conv1.shape[2])
    combine_filters = Conv3D(num_filter[1], (1, conv_x_size, conv_x_size), strides=(1, 1, 1), name='LPTC',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=regularizers.l1(reg_val))(conv1_d)
    #combine_filters = BatchNormalization()(combine_filters)
    combine_filters_a = Activation('relu')(combine_filters)
    combine_filters_d = Dropout(d_rate)(combine_filters_a)

    # behavior
    behavior = Conv3D(3, (1, 1, 1), strides=(1, 1, 1), name='behavior',
                      kernel_initializer=glorot_uniform(seed=None))(combine_filters_d)

    # Create model
    model = Model(inputs=image_in, outputs=behavior, name='ln_model')

    return model, pad_x, pad_y, pad_t, learning_rate, batch_size


def hrc_model(input_shape=(11, 9, 1), filter_shape=(21, 2), num_hrc=1, sum_over_space=True):
    # set the learning rate that works for this model
    learning_rate = 0.001 * 1
    batch_size = np.power(2, 6)

    # output the amount that this model will reduce the space and time variable by
    pad_x = int((filter_shape[1] - 1) / 2)
    pad_t = int((filter_shape[0] - 1) / 2)

    # Define the input as a tensor with shape input_shape
    model_input = Input(input_shape)

    left_in = Conv2D(num_hrc, filter_shape, strides=(1, 1), name='conv1',
                       kernel_initializer=glorot_uniform(seed=None))(model_input)
    right_in = Conv2D(num_hrc, filter_shape, strides=(1, 1), name='conv2',
                      kernel_initializer=glorot_uniform(seed=None))(model_input)

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    multiply_layer = multiply([left_in, right_in])

    # full_reich = unit1_multiply

    # combine all the correlators
    # conv_x_size = int(x_layer2.shape[2])
    conv_x_size = 1
    combine_corr = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='x_out',
                   kernel_initializer=glorot_uniform(seed=None))(multiply_layer)

    if sum_over_space:
        sum_reich = sum_layer(combine_corr)
    else:
        sum_reich = combine_corr

    # Create model
    model = Model(inputs=model_input, outputs=sum_reich, name='ReichCorr')

    return model, pad_x, pad_t, learning_rate, batch_size

