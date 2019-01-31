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


def r2(y_true, y_pred, ind):
    y_t = y_true[:, :, :, :, ind:ind+1]
    y_p = y_pred[:, :, :, :, ind:ind+1]
    num = K.sum(K.square(y_p - y_t), axis=1, keepdims=True)
    denom = K.sum(K.square(y_t - K.mean(y_t)), axis=1, keepdims=True)
    r2_value = 1 - K.exp(K.mean(K.log(num/denom)))
    return r2_value

"""
def r2(y_true, y_pred, ind):
    r2_value = 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))
    return r2_value
"""

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


def hrc_model(input_shape, filter_shape, num_filter=(8, 6)):
    learning_rate = 0.001 * 0.1
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
    conv1_l = Conv3D(num_filter[0], filter_shape, strides=(1, 1, 1), name='T4_T5_l',
                   kernel_initializer=glorot_uniform(seed=None),
                   kernel_regularizer=regularizers.l1(reg_val))(image_in)
    # conv1_l = BatchNormalization()(conv1)
    conv1_l_d = Dropout(d_rate)(conv1_l)

    conv1_r = Conv3D(num_filter[0], filter_shape, strides=(1, 1, 1), name='T4_T5_r',
                   kernel_initializer=glorot_uniform(seed=None),
                   kernel_regularizer=regularizers.l1(reg_val))(image_in)
    # conv1_r = BatchNormalization()(conv1_r)
    conv1_r_d = Dropout(d_rate)(conv1_r)

    multiply_step = multiply([conv1_l_d, conv1_r_d])

    # LPTCs
    # convolution that takes up all of space
    conv_x_size = int(multiply_step.shape[2])
    combine_filters = Conv3D(num_filter[1], (1, conv_x_size, conv_x_size), strides=(1, 1, 1), name='LPTC',
                             kernel_initializer=glorot_uniform(seed=None),
                             kernel_regularizer=regularizers.l1(reg_val))(multiply_step)
    # combine_filters = BatchNormalization()(combine_filters)
    combine_filters_a = Activation('relu')(combine_filters)
    combine_filters_d = Dropout(d_rate)(combine_filters_a)

    # behavior
    behavior = Conv3D(3, (1, 1, 1), strides=(1, 1, 1), name='behavior',
                      kernel_initializer=glorot_uniform(seed=None))(combine_filters_d)

    # Create model
    model = Model(inputs=image_in, outputs=behavior, name='ln_model')

    return model, pad_x, pad_y, pad_t, learning_rate, batch_size

