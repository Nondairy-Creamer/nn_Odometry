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
data_set_folder = 'G:\\My Drive\\data_sets\\nn_Odometry\\'
# data_set_name = 'dataset-outdoors3_512_16.tar'
data_set_name = 'dataset-corridor4_512_16.tar'

path = data_set_folder + 'natural_images\\' + data_set_name

# load in data set
train_in, train_out, dev_in, dev_out, test_in, test_out, sample_freq, phase_step = md.load_data_rr(path)

# get sample rate of dataset

filter_time = 0.1  # s
filter_space = 100  # degrees

filter_indicies_t = int(np.ceil(filter_time*sample_freq)+1)
filter_indicies_x = int(np.ceil(filter_space/phase_step)+1)

# filters must have odd length
assert(np.mod(filter_indicies_t, 2) == 1)
assert(np.mod(filter_indicies_x, 2) == 1)

# intiialize model
m, size_t, size_x, n_c = train_in.shape
sum_over_space = False
# model, pad_x, pad_t, learning_rate, batch_size = md.ln_model(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=2, sum_over_space=sum_over_space)
# model, pad_x, pad_t, learning_rate, batch_size = md.ln_model_deep(input_shape=(size_t, size_x, n_c), filter_shape=((filter_indicies_t, filter_indicies_x), (filter_indicies_t, filter_indicies_x)), num_filter=(16, 4))
# model, pad_x, pad_t, learning_rate, batch_size = md.hrc_model(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_hrc=1, sum_over_space=sum_over_space)
# model, pad_x, pad_t, learning_rate, batch_size = md.hrc_model_sep(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_hrc=1, sum_over_space=sum_over_space)
# model, pad_x, pad_t, learning_rate, batch_size = md.hrc_model_no_flip(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_hrc=4, sum_over_space=sum_over_space)
model, pad_x, pad_t, learning_rate, batch_size = md.ln_model_no_flip(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=4, sum_over_space=sum_over_space)
# model, pad_x, pad_t, learning_rate, batch_size = md.ln_model_no_flip_deep(input_shape=(size_t, size_x, n_c), filter_shape=(filter_indicies_t, filter_indicies_x), num_filter=2, sum_over_space=sum_over_space)

# format y data to fit with output
if sum_over_space:
    train_out = train_out[:, 0:-1 - 2 * pad_t + 1, :, :]
    dev_out = dev_out[:, 0:-1 - 2 * pad_t + 1, :, :]
    test_out = test_out[:, 0:-1 - 2 * pad_t + 1, :, :]
else:
    # repeat y data to fit output conv size
    train_out = np.tile(train_out, (1, 1, size_x, 1))
    dev_out = np.tile(dev_out, (1, 1, size_x, 1))
    test_out = np.tile(test_out, (1, 1, size_x, 1))

    train_out = train_out[:, 0:-1-2*pad_t+1, pad_x:-1-pad_x+1, :]
    dev_out = dev_out[:, 0:-1-2*pad_t+1, pad_x:-1-pad_x+1, :]
    test_out = test_out[:, 0:-1-2*pad_t+1, pad_x:-1-pad_x+1, :]

# normalize images
train_in = train_in/np.std(train_in, axis=(1, 2), keepdims=True)
dev_in = dev_in/np.std(dev_in, axis=(1, 2), keepdims=True)
test_in = test_in/np.std(test_in, axis=(1, 2), keepdims=True)

train_out = train_out/np.std(train_out, axis=(1, 2), keepdims=True)
dev_out = dev_out/np.std(dev_out, axis=(1, 2), keepdims=True)
test_out = test_out/np.std(test_out, axis=(1, 2), keepdims=True)

# set up the model and fit it
t = time.time()
adamOpt = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[md.r2])
hist = model.fit(train_in, train_out, verbose=2, epochs=50, batch_size=batch_size, validation_data=(dev_in, dev_out))
elapsed = time.time() - t

# grab the loss and R2 over time
#model.save('kerasModel_' + str(num_filt) + 'Filt' + '.h5')
loss = hist.history['loss']
val_loss = hist.history['val_loss']
r2 = hist.history['r2']
val_r2 = hist.history['val_r2']

print('model took ' + str(elapsed) + 's to train')

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

weights_name = 'weights_' + str(filter_time) + 'filterTime_' + str(filter_space) + 'filterSpace_' + str(int(sample_freq)) + 'sampleFreq_' + str(int(phase_step)) + 'phaseStep'
weights_name = "-".join(weights_name.split("."))
save_path = data_set_folder + '\\saved_parameters\\' + weights_name
sio.savemat(save_path, imageDict)
# plot_model(model, to_file='kerasModel_structure.png')
