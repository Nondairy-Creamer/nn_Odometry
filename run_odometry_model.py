import odometry_model as md
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
import scipy.io as sio
import tensorflow as tf

# define the input path
# data set location
data_set_folder = 'G:\\My Drive\\data_sets\\nn_Odometry\\composite_dataset\\'
data_set_name = 'image_dataset_filtered_concat.h5'
path = data_set_folder + data_set_name

# load in data set
train_set, dev_set, test_set, train_ans, dev_ans, test_ans, sample_freq, phase_step = md.load_data_rr(path)

# get sample rate of dataset

filter_time = 1  # s
filter_space = 30  # degrees

filter_indicies_t = int(np.ceil(filter_time*sample_freq)+1)
filter_indicies_y = int(np.ceil(filter_space/phase_step)+1)
filter_indicies_x = int(np.ceil(filter_space/phase_step)+1)

# filters must have odd length
assert(np.mod(filter_indicies_t, 2) == 1)
assert(np.mod(filter_indicies_x, 2) == 1)
assert(np.mod(filter_indicies_y, 2) == 1)

# intiialize model
m, size_t, size_y, size_x, n_c = train_set.shape
model, pad_x, pad_y, pad_t, learning_rate, batch_size = md.ln_model(input_shape=(size_t, size_y, size_x, n_c),
                                                                    filter_shape=(filter_indicies_t, filter_indicies_y, filter_indicies_x),
                                                                    num_filter=(4, 6))
# model, pad_x, pad_t, learning_rate, batch_size = md.hrc_model(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=4, sum_over_space=sum_over_space)

# format y data to fit with output
train_ans = train_ans[:, 0:-1 - 2 * pad_t + 1, :, :, 0:3]
dev_ans = dev_ans[:, 0:-1 - 2 * pad_t + 1, :, :, 0:3]
test_ans = test_ans[:, 0:-1 - 2 * pad_t + 1, :, :, 0:3]

train_ans[np.isnan(train_ans)] = 0
dev_ans[np.isnan(dev_ans)] = 0
test_ans[np.isnan(test_ans)] = 0

# normalize images
train_set = train_set/np.std(train_set, axis=(1, 2), keepdims=True)
dev_set = dev_set/np.std(dev_set, axis=(1, 2), keepdims=True)
test_set = test_set/np.std(test_set, axis=(1, 2), keepdims=True)

# set up the model and fit it
t = time.time()
adamOpt = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[md.r2])
hist = model.fit(train_set, train_ans, verbose=2, epochs=500, batch_size=batch_size, validation_data=(dev_set, dev_ans))
elapsed = time.time() - t

# grab the loss and R2 over time
#model.save('kerasModel_' + str(num_filt) + 'Filt' + '.h5')
loss = hist.history['loss']
val_loss = hist.history['val_loss']
r2 = hist.history['r2']
val_r2 = hist.history['val_r2']

print('model took ' + str(elapsed) + 's to train')

# plot loss and R2
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val loss'])

plt.subplot(1, 2, 2)
plt.plot(r2)
plt.plot(val_r2)
plt.legend(['r2', 'val r2'])
plt.ylim((0, 1))
plt.show()

imageDict = {}
ww = 0

for l in model.layers:
    all_weights = l.get_weights()

    ww += 1

    if len(all_weights) > 0:
        weights = all_weights[0]
        if len(all_weights) == 2:
            biases = all_weights[1]
        else:
            biases = [0]

        imageDict["weight" + str(ww)] = weights
        imageDict["biases" + str(ww)] = biases

        maxAbsW1 = np.max(np.abs(weights))

        for c in range(weights.shape[2]):
            plt.figure()
            for w in range(weights.shape[3]):
                plt.subplot(2, int(np.ceil(weights.shape[3]/2)), w+1)
                img = plt.imshow(weights[:, :, c, w])
                plt.clim(-maxAbsW1, maxAbsW1)
                plt.axis('off')
                img.set_cmap('RdBu_r')
                # plt.colorbar()

        plt.figure()
        x = np.arange(len(biases))
        plt.scatter(x, biases)

weights_name = 'weights_' + str(filter_time) + 'filterTime_' + str(filter_space) + 'filterSpace_' + str(int(sample_freq)) + 'sampleFreq_' + str(int(phase_step)) + 'phaseStep'
weights_name = "-".join(weights_name.split("."))
save_path = data_set_folder + '\\saved_parameters\\' + weights_name
sio.savemat(save_path, imageDict)
# plot_model(model, to_file='kerasModel_structure.png')
