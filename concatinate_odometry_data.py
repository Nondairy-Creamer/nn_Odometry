import numpy as np
import h5py
import scipy.io as sio
import os

# define the input path
# data set location
data_set_folder = 'G:\\My Drive\\data_sets\\nn_Odometry\\natural_images\\'
composite_folder = 'G:\\My Drive\\data_sets\\nn_Odometry\\composite_dataset\\'
data_set_name = sorted(os.listdir(data_set_folder))[5:11]

# get initial size

data_in = h5py.File(data_set_folder + data_set_name[0] + '\\image_dataset_filtered.h5', 'r')

train_set = data_in['train_set'][:]
dev_set = data_in['dev_set'][:]
test_set = data_in['test_set'][:]

train_ans = data_in['train_ans'][:]
dev_ans = data_in['dev_ans'][:]
test_ans = data_in['test_ans'][:]

sample_freq = data_in['sample_freq'][:]
phase_step = data_in['phase_step'][:]

data_in.close()
for data_set_in in data_set_name:
    data_in = h5py.File(data_set_folder + data_set_in + '\\image_dataset_filtered.h5', 'r')

    train_set = np.concatenate((train_set, data_in['train_set'][:]))
    dev_set = np.concatenate((dev_set, data_in['dev_set'][:]))
    test_set = np.concatenate((test_set, data_in['test_set'][:]))

    train_ans = np.concatenate((train_ans, data_in['train_ans'][:]))
    dev_ans = np.concatenate((dev_ans, data_in['dev_ans'][:]))
    test_ans = np.concatenate((test_ans, data_in['test_ans'][:]))

    sample_freq = data_in['sample_freq'][:]
    phase_step = data_in['phase_step'][:]

    data_in.close()

# save the new data
h5f = h5py.File(composite_folder + 'image_dataset_filtered_concat.h5', 'w')

h5f.create_dataset('train_set', data=train_set)
h5f.create_dataset('dev_set', data=dev_set)
h5f.create_dataset('test_set', data=test_set)

h5f.create_dataset('train_ans', data=train_ans)
h5f.create_dataset('dev_ans', data=dev_ans)
h5f.create_dataset('test_ans', data=test_ans)

h5f.create_dataset('sample_freq', data=sample_freq)
h5f.create_dataset('phase_step', data=phase_step)

h5f.close()
sio.savemat(composite_folder + 'image_dataset_filtered_concat_temp.mat', {'train_set': train_set,
                                                                          'dev_set': dev_set,
                                                                          'test_set': test_set,
                                                                          'train_ans': train_ans,
                                                                          'dev_ans': dev_ans,
                                                                          'test_ans': test_ans,
                                                                          'sample_freq': sample_freq,
                                                                          'phase_step': phase_step})

