import tarfile
import os
import imageio as io
import numpy as np
import h5py
import scipy.io as sio

# force extraction of tar
force_tar_extract = False
force_image_extract = True
# define the input path
# data set location
data_set_folder = 'G:\\My Drive\\data_sets\\nn_Odometry\\natural_images\\'
# data_set_name = 'dataset-corridor1_512_16'
data_set_name = 'dataset-corridor4_512_16'
data_set_name = sorted(os.listdir(data_set_folder))[11:12]
# data_set_name = data_set_name[3:]
total_num_folder = len(data_set_name)
folder_num = 0

for data_set_in in data_set_name:
    folder_num += 1

    extract_path = data_set_folder
    tar_path = data_set_folder + data_set_in + '.tar'
    folder_path = data_set_folder + data_set_in + '\\'
    image_path = folder_path + 'mav0\\cam0\\data\\'
    image_data_path = folder_path + 'mav0\\cam0\\data.csv'
    motion_path = folder_path + 'mav0\\imu0\\data.csv'

    if not os.path.isfile(folder_path + 'image_dataset.h5') or force_image_extract:
        #try:

        # if the tar doesn't exist or we're forcing it, extract it
        if not os.path.isdir(folder_path) or force_tar_extract:
            dataIn = tarfile.open(tar_path)
            dataIn.extractall(extract_path)
            dataIn.close()

        # read in the images
        image_size = (512, 512)
        num_files = len(os.listdir(image_path))
        images = np.zeros((num_files, image_size[0], image_size[1]), dtype='uint16')
        file_ind = 0
        image_timestamp = np.genfromtxt(image_data_path, delimiter=',', skip_header=1)
        image_timestamp = image_timestamp[:, 0]
        total_files = len(os.listdir(image_path))

        for file in sorted(os.listdir(image_path)):
            if file.endswith('.png'):
                images[file_ind, :, :] = io.imread(image_path + file)
                file_ind += 1

            if np.mod(file_ind, 10) == 0:
                print('folder ' + str(folder_num) + '/' + str(total_num_folder))
                print(str(file_ind) + '/' + str(total_files) + ' files completed')

        # read in the motion data
        motion_data = np.genfromtxt(motion_path, delimiter=',', skip_header=1)
        # drop time stamps
        motion_timestamp = motion_data[:, 0]
        motion_data = motion_data[:, 1:]
        # convert motion data to velocity
        # currently unsure if its in velocity or acceleration
        # motion_data[:, 3:] = np.cumsum(motion_data[:, 3:], axis=0)


        # break up into train, dev, and test sets
        dev_frac = 0.05
        test_frac = dev_frac

        # I believe images have a FOV of 106 degrees
        # data is taken at 20 hz
        # split data set up into 5s chunks or 101 indicies
        num_time = 5  # s
        sample_freq = 20  # Hz
        fov = 106  # degrees
        phase_step = fov/image_size[0]

        movie_size = images.shape
        motion_size = motion_data.shape
        chunk_size_ind = num_time*sample_freq
        num_chunks = int(np.floor(movie_size[0]/chunk_size_ind))

        # compress motion data to image size
        last_time_stamp = 0
        motion_data_ave = np.zeros((len(image_timestamp), motion_size[1]))
        for m_ind in range(len(image_timestamp)):
            indicies_since_last_time_stamp = (motion_timestamp <= image_timestamp[m_ind]) & (motion_timestamp > last_time_stamp)
            motion_data_ave[m_ind, :] = np.mean(motion_data[indicies_since_last_time_stamp, :], axis=0)
            last_time_stamp = image_timestamp[m_ind]

        # delete motion data
        images = np.reshape(images[0:num_chunks*chunk_size_ind], [num_chunks, chunk_size_ind, movie_size[1], movie_size[2]])

        images = np.expand_dims(images, axis=4)

        motion_data_ave = np.reshape(motion_data_ave[0:num_chunks*chunk_size_ind], [num_chunks, chunk_size_ind, motion_size[1]])

        motion_data_ave = np.expand_dims(motion_data_ave, axis=3)
        motion_data_ave = np.expand_dims(motion_data_ave, axis=4)
        motion_data_ave = np.transpose(motion_data_ave, axes=[0, 1, 3, 4, 2])

        chunk_order = np.random.permutation(num_chunks)
        dev_num = int(np.ceil(num_chunks*dev_frac))
        test_num = int(np.ceil(num_chunks*test_frac))
        train_num = int(num_chunks - dev_num - test_num)

        train_ind = chunk_order[slice(0, train_num)]
        dev_ind = chunk_order[slice(train_num, train_num+dev_num)]
        test_ind = chunk_order[slice(train_num+dev_num, train_num+dev_num+test_num)]

        # save the data
        h5f = h5py.File(folder_path + 'image_dataset.h5', 'w')

        h5f.create_dataset('train_set', data=images[train_ind, :, :, :, :])
        h5f.create_dataset('dev_set', data=images[dev_ind, :, :, :, :])
        h5f.create_dataset('test_set', data=images[test_ind, :, :, :, :])

        h5f.create_dataset('train_ans', data=motion_data_ave[train_ind, :, :, :, :])
        h5f.create_dataset('dev_ans', data=motion_data_ave[dev_ind, :, :, :, :])
        h5f.create_dataset('test_ans', data=motion_data_ave[test_ind, :, :, :, :])

        h5f.create_dataset('sample_freq', data=[sample_freq])
        h5f.create_dataset('phase_step', data=[phase_step])

        h5f.close()
        sio.savemat(folder_path + 'image_dataset_temp', {'dev_set': images[dev_ind, :, :, :, :], 'dev_ans': motion_data_ave[dev_ind, :, :, :, :]})

        #except:
            #print('error:' + data_set_in)

