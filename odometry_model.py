from tensorflow.keras.layers import multiply, concatenate, LSTM, Dense, Input, Activation, BatchNormalization, Conv2D, subtract, add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
#import tensorflow.keras as TFK
import tensorflow.keras.backend as K
import h5py as h
import numpy as np
from tensorflow.keras.layers import *
import scipy.io as sio
import tarfile


def load_data_rr(path):
    data_in = np.load(path + 'image_dataset')

    train_set = data_in['train_set']
    dev_set = data_in['dev_set']
    test_set = data_in['test_set']

    train_ans = data_in['train_ans']
    dev_ans = data_in['dev_ans']
    test_ans = data_in['test_ans']

    sample_freq = data_in['sample_freq']
    phase_step = data_in['phase_step']

    return train_set, dev_set, test_set, train_ans, dev_ans, test_ans, sample_freq, phase_step


def r2(y_true, y_pred):
    r2_value = 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))
    return r2_value


def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2, sum_over_space=True):
    learning_rate = 0.001*1
    batch_size = np.power(2, 5)

    # Define the input as a tensor with shape input_shape
    image_in = Input(input_shape)

    pad_x = int((filter_shape[1] - 1) / 2)
    pad_t = int((filter_shape[0] - 1) / 2)

    conv1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=None), activation='relu')(image_in)

    # make sum layer
    sum_layer = Lambda(lambda lam: K.sum(lam, axis=2, keepdims=True))

    # conv_x_size = int(x_layer2.shape[2])
    conv_x_size = 1
    combine_filters = Conv2D(1, (1, conv_x_size), strides=(1, 1), name='conv2', kernel_initializer=glorot_uniform(seed=None))(conv1)

    if sum_over_space:
        averaged_space = sum_layer(combine_filters)
    else:
        averaged_space = combine_filters

    # Create model
    model = Model(inputs=image_in, outputs=averaged_space, name='SimpleMotion')

    return model, pad_x, pad_t, learning_rate, batch_size


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

