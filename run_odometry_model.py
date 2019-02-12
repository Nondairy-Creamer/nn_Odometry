import odometry_model as md
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
import scipy.io as sio
from tensorflow.keras.layers import Lambda
import tensorflow as tf

# define the input path
# data set location
data_set_folder = 'G:\\My Drive\\data_sets\\nn_Odometry\\'
data_set_name = 'image_dataset_filtered_concat.h5'
path = data_set_folder + 'composite_dataset\\' + data_set_name
learn_trans = False

# load in data set
train_set, dev_set, test_set, train_ans, dev_ans, test_ans, sample_freq, phase_step = md.load_data_rr(path)

# get sample rate of dataset

filter_time = 0.5  # s
filter_space = 50  # degrees

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
#model, pad_x, pad_y, pad_t, learning_rate, batch_size = md.hrc_model(input_shape=(size_t, size_y, size_x, n_c),
#                                                                    filter_shape=(filter_indicies_t, filter_indicies_y, filter_indicies_x),
#                                                                    num_filter=(4, 6))
# format y data to fit with output
if learn_trans:
    train_ans = train_ans[:, 0:-1 - 2 * pad_t + 1, :, :, 0:]
    dev_ans = dev_ans[:, 0:-1 - 2 * pad_t + 1, :, :, 0:]
    test_ans = test_ans[:, 0:-1 - 2 * pad_t + 1, :, :, 0:]
else:
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


# generate metrics
def r2_1(x, y):
    return md.r2(x, y, 0)
def r2_2(x, y):
    return md.r2(x, y, 1)
def r2_3(x, y):
    return md.r2(x, y, 2)
def r2_4(x, y):
    return md.r2(x, y, 3)
def r2_5(x, y):
    return md.r2(x, y, 4)
def r2_6(x, y):
    return md.r2(x, y, 5)

# set up the model and fit it
t = time.time()
adamOpt = optimizers.Adam(lr=learning_rate)
if learn_trans:
    model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[r2_1, r2_2, r2_3, r2_4, r2_5, r2_6])
else:
    model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[r2_1, r2_2, r2_3])

hist = model.fit(train_set, train_ans, verbose=2, epochs=1000, batch_size=batch_size, validation_data=(dev_set, dev_ans))
elapsed = time.time() - t

# grab the loss and R2 over time
#model.save('kerasModel_' + str(num_filt) + 'Filt' + '.h5')
loss = hist.history['loss']
val_loss = hist.history['val_loss']

if learn_trans:
    r2 = np.zeros((6, len(loss)))
    val_r2 = np.zeros((6, len(loss)))

    r2[0, :] = hist.history['r2_1']
    r2[1, :] = hist.history['r2_2']
    r2[2, :] = hist.history['r2_3']
    r2[3, :] = hist.history['r2_4']
    r2[4, :] = hist.history['r2_5']
    r2[5, :] = hist.history['r2_6']

    val_r2[0, :] = hist.history['val_r2_1']
    val_r2[1, :] = hist.history['val_r2_2']
    val_r2[2, :] = hist.history['val_r2_3']
    val_r2[3, :] = hist.history['val_r2_4']
    val_r2[4, :] = hist.history['val_r2_5']
    val_r2[5, :] = hist.history['val_r2_6']
else:
    r2 = np.zeros((3, len(loss)))
    val_r2 = np.zeros((3, len(loss)))

    r2[0, :] = hist.history['r2_1']
    r2[1, :] = hist.history['r2_2']
    r2[2, :] = hist.history['r2_3']

    val_r2[0, :] = hist.history['val_r2_1']
    val_r2[1, :] = hist.history['val_r2_2']
    val_r2[2, :] = hist.history['val_r2_3']

print('model took ' + str(elapsed) + 's to train')



image_dict = {}
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

        image_dict["weight" + str(ww)] = weights
        image_dict["biases" + str(ww)] = biases

        maxAbsW1 = np.max(np.abs(weights))

        for c_out in range(weights.shape[4]):
            for c_in in range(weights.shape[3]):
                plt.figure()

                num_plot_x = int(np.ceil(np.sqrt(weights.shape[0])))
                num_plot_y = int(np.ceil(weights.shape[0]/num_plot_x))

                fig, axs = plt.subplots(num_plot_y, num_plot_x, constrained_layout=True)
                fig.suptitle('c_out ' + str(c_out+1) + ', c_in ' + str(c_in+1), fontsize=16)

                for t_ind in range(weights.shape[0]):
                    fig = plt.subplot(num_plot_y, num_plot_x, t_ind+1)
                    img = plt.imshow(weights[t_ind, :, :, c_in, c_out])
                    plt.clim(-maxAbsW1, maxAbsW1)
                    plt.axis('off')
                    img.set_cmap('RdBu_r')
                    # plt.colorbar()

# plot loss and R2
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val loss'])

plt.subplot(1, 2, 2)
plt.plot(r2.T)
plt.plot(val_r2.T)
plt.legend(['r2', 'val r2'])
plt.ylim((0, 1))

plt.show()

weights_name = 'weights_' + str(filter_time) + 'filterTime_' + str(filter_space) + 'filterSpace_' + str(int(sample_freq)) + 'sampleFreq_' + str(int(phase_step)) + 'phaseStep'
weights_name = "-".join(weights_name.split("."))
save_path = data_set_folder + 'saved_parameters\\' + weights_name
sio.savemat(save_path, image_dict)
# plot_model(model, to_file='kerasModel_structure.png')
