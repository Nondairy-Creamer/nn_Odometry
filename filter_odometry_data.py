import numpy as np
import scipy as sc
import h5py
from scipy import ndimage
import scipy.io as sio
import tensorflow as tf
import os

def matt_conv_3d(x_in, h_in, strides):
    stim = tf.placeholder('float32', [None, None, None, None, 1])
    h = tf.placeholder('float32', [None, None, None, 1, 1])

    conv = tf.nn.conv3d(stim, h, strides, 'VALID')

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    out = sess.run(conv, feed_dict={stim: x_in, h: h_in})

    return out

# force extraction of tar
force_tar_extract = False

# define the input path
# data set location
data_set_folder = 'G:\\My Drive\\data_sets\\nn_Odometry\\natural_images\\'
# data_set_name = 'dataset-corridor1_512_16'
# data_set_name = 'dataset-corridor4_512_16'
data_set_name = sorted(os.listdir(data_set_folder))[0:11]
total_num_folder = len(data_set_name)

for data_set_in in data_set_name:
    #try:
        data_in = h5py.File(data_set_folder + data_set_in + '\\image_dataset.h5', 'r')

        train_set = data_in['train_set'][:]
        dev_set = data_in['dev_set'][:]
        test_set = data_in['test_set'][:]

        train_ans = data_in['train_ans'][:]
        dev_ans = data_in['dev_ans'][:]
        test_ans = data_in['test_ans'][:]

        sample_freq = data_in['sample_freq'][:]
        phase_step = data_in['phase_step'][:]

        data_in.close()

        movie_size = train_set.shape

        num_std = 3
        std_small = 2.5  # degrees
        std_big = 10  # degrees

        x = np.arange(-num_std*std_big/2, num_std*std_big/2, phase_step)

        # make the filters gaussians
        filt_x_small = 1/(np.sqrt(2*np.pi)*std_small)*np.exp(-np.square(x)/(2*np.square(std_small)))
        filt_xy_small = np.outer(filt_x_small, filt_x_small)
        filt_xy_small = np.expand_dims(filt_xy_small, axis=2)
        filt_xy_small = np.expand_dims(filt_xy_small, axis=3)
        filt_xy_small = np.expand_dims(filt_xy_small, axis=4)
        filt_xy_small = np.transpose(filt_xy_small, axes=[2, 0, 1, 3, 4])
        filt_xy_small = filt_xy_small.astype('float32')

        filt_x_big = 1/(np.sqrt(2*np.pi)*std_big)*np.exp(-np.square(x)/(2*np.square(std_big)))
        filt_xy_big = np.outer(filt_x_big, filt_x_big)
        filt_xy_big = np.expand_dims(filt_xy_big, axis=2)
        filt_xy_big = np.expand_dims(filt_xy_big, axis=3)
        filt_xy_big = np.expand_dims(filt_xy_big, axis=4)
        filt_xy_big = np.transpose(filt_xy_big, axes=[2, 0, 1, 3, 4])
        filt_xy_big = filt_xy_big.astype('float32')

        step_size = int(np.round(2*std_small/phase_step))
        s = [1, 1, step_size, step_size, 1]

        # filter the images
        dev_set_small = matt_conv_3d(dev_set, filt_xy_small, s)
        dev_set_big = matt_conv_3d(dev_set, filt_xy_big, s)

        test_set_small = matt_conv_3d(test_set, filt_xy_small, s)
        test_set_big = matt_conv_3d(test_set, filt_xy_big, s)

        train_set_size = [train_set.shape[0],
                          int(np.floor((train_set.shape[1]-filt_xy_big.shape[0])/s[1]))+1,
                          int(np.floor((train_set.shape[2]-filt_xy_big.shape[1])/s[2]))+1,
                          int(np.floor((train_set.shape[3]-filt_xy_big.shape[2])/s[3]))+1,
                          train_set.shape[4]]

        train_set_small = np.zeros(train_set_size, dtype='float32')
        train_set_big = np.zeros(train_set_size, dtype='float32')
        for mm in range(train_set.shape[0]):
            train_set_small[[mm], :, :, :, :] = matt_conv_3d(train_set[[mm], :, :, :, :], filt_xy_small, s)
            train_set_big[[mm], :, :, :, :] = matt_conv_3d(train_set[[mm], :, :, :, :], filt_xy_big, s)

        # turn in to contrast
        train_set = (train_set_small - train_set_big)/train_set_big
        dev_set = (dev_set_small - dev_set_big)/dev_set_big
        test_set = (test_set_small - test_set_big)/test_set_big

        train_set = np.reshape(train_set, (train_set.shape[1]))

        # save the new data
        h5f = h5py.File(data_set_folder + data_set_in + '\\image_dataset_filtered.h5', 'w')

        h5f.create_dataset('train_set', data=train_set)
        h5f.create_dataset('dev_set', data=dev_set)
        h5f.create_dataset('test_set', data=test_set)

        h5f.create_dataset('train_ans', data=train_ans)
        h5f.create_dataset('dev_ans', data=dev_ans)
        h5f.create_dataset('test_ans', data=test_ans)

        h5f.create_dataset('sample_freq', data=sample_freq)
        h5f.create_dataset('phase_step', data=[2*std_small])

        h5f.close()
        sio.savemat(data_set_folder + data_set_in + '\\image_dataset_filtered_temp.mat', {'dev_set': dev_set, 'dev_ans': dev_ans})

    #except:
        #print('error in:' + data_set_in)

