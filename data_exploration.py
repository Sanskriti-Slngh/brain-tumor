import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

img = nib.load('G:\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001/BraTS20_Training_001_seg.nii')
for i in range(img.dataobj.shape[-1]):
    image = img.dataobj[:,:, i]
    if np.count_nonzero(image) != 0:
        print("This will work")

#data = img.get_fdata()
data = img.astype(np.int8)

fig, axs = plt.subplots(12, 12)
print (axs.shape)
for i in range(144):
    axs[i//12, i%12].imshow(data, cmap="gray")

plt.plot()
plt.show()

print(data.shape)
print(data.dtype)