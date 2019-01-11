from tensorflow.keras.models import load_model
import modelKeras as md
import numpy as np
import matplotlib.pyplot as plt

model4 = load_model('kerasModel_1Filt.h5', custom_objects={'r2': md.r2})

weights = model4.get_weights()

W1 = weights[0]
b1 = weights[1]

W2 = weights[2]
b2 = weights[3]

maxAbsW1 = np.max(np.abs(W1))

num_filt = W1.shape[3]
rows = round(np.sqrt(num_filt))
cols = np.ceil(np.sqrt(num_filt))

plt.figure(1)
for w in range(num_filt):
    plt.subplot(rows, cols, w+1)
    img = plt.imshow(W1[:, :, 0, w])
    plt.clim(-maxAbsW1, maxAbsW1)
    plt.colorbar()
    plt.axis('off')
    img.set_cmap('RdBu_r')

plt.show()

print('b1 = ' + str(b1))
print('W2 = ' + str(W2))
print('b2 = ' + str(b2))

