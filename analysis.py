import numpy as np
import matplotlib.pyplot as plt

a = np.load("./analysis/Fold1_Testing_A.npy")
b = np.load("./analysis/Fold1_Training_A.npy")


print(a.shape, b.shape)

for i in range(a.shape[0]):
    plt.imshow(a[i, :, :])
    plt.show()