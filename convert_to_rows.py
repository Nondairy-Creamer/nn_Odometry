import numpy as np
import h5py
import scipy.io as sio

data_set_folder = 'G:\\My Drive\\data_sets\\nn_Odometry\\'
data_set_name = 'image_dataset_filtered_concat.h5'
path_in = data_set_folder + 'composite_dataset\\' + data_set_name
path_out = data_set_folder + 'composite_dataset\\' + 'image_dataset_filtered_rows.h5'

data_in = h5py.File(path_in, 'r')

train_set = data_in['train_set'][:]
dev_set = data_in['dev_set'][:]
test_set = data_in['test_set'][:]

train_ans = data_in['train_ans'][:]
dev_ans = data_in['dev_ans'][:]
test_ans = data_in['test_ans'][:]

sample_freq = data_in['sample_freq'][:]
phase_step = data_in['phase_step'][:]

data_in.close()

# reshape the data so that each y row is now a different example
full_set = np.concatenate((train_set, dev_set, test_set))
full_ans = np.concatenate((train_ans, dev_ans, test_ans))
full_ans = full_ans[:, :, :, :, 2:3]
ans_rep = full_set.shape[2]

full_set = np.transpose(full_set, (0, 2, 1, 3, 4))
full_set = np.reshape(full_set, (full_set.shape[0]*full_set.shape[1], full_set.shape[2], full_set.shape[3], full_set.shape[4]))

full_ans = np.transpose(full_ans, (0, 2, 1, 3, 4))
full_ans = np.repeat(full_ans, ans_rep, axis=1)
full_ans = np.reshape(full_ans, (full_ans.shape[0]*full_ans.shape[1], full_ans.shape[2], full_ans.shape[3], full_ans.shape[4]))

# find out the cut offs for the train/dev/test sets
dev_frac = 0.05
test_frac = dev_frac
dev_num = int(np.ceil(full_set.shape[0] * dev_frac))
test_num = int(np.ceil(full_set.shape[0] * test_frac))
train_num = int(full_set.shape[0] - dev_num - test_num)
rand_ind = np.random.permutation(full_set.shape[0])

train_ind = rand_ind[slice(0, train_num)]
dev_ind = rand_ind[slice(train_num, train_num+dev_num)]
test_ind = rand_ind[slice(train_num+dev_num, train_num+dev_num+test_num)]

train_set = full_set[train_ind, :, :, :]
dev_set = full_set[dev_ind, :, :, :]
test_set = full_set[test_ind, :, :, :]

train_ans = full_ans[train_ind, :, :, :]
dev_ans = full_ans[dev_ind, :, :, :]
test_ans = full_ans[test_ind, :, :, :]

# save the new data
h5f = h5py.File(path_out, 'w')

h5f.create_dataset('train_set', data=train_set)
h5f.create_dataset('dev_set', data=dev_set)
h5f.create_dataset('test_set', data=test_set)

h5f.create_dataset('train_ans', data=train_ans)
h5f.create_dataset('dev_ans', data=dev_ans)
h5f.create_dataset('test_ans', data=test_ans)

h5f.create_dataset('sample_freq', data=sample_freq)
h5f.create_dataset('phase_step', data=phase_step)

h5f.close()
sio.savemat(data_set_folder + 'composite_dataset\\' + '\\image_dataset_filtered_rows_temp.mat', {'dev_set': dev_set, 'dev_ans': dev_ans})

