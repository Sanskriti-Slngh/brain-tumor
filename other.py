import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

directory = 'C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/'

img = nib.load("C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_079/BraTS20_Training_079_seg.nii.gz")
imgs = img.dataobj[:,:,132]
same = imgs
x, y, width, height = cv2.boundingRect(same)
print(imgs)
plt.imshow(imgs)
plt.show()

flt = nib.load("C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_089/BraTS20_Training_089_seg.nii.gz")
flts = flt.dataobj[:,:,132]
x, y, width, height = cv2.boundingRect(flts)
same_2 = flts
print(flts)
plt.imshow(flts)
plt.show()

print((imgs==flts).all())
exit()


for index in range(369):
    for i in range(2):
        if index+1 == 89:
            index = index + 1
        file = '%sBraTS20_Training_%s/BraTS20_Training_%s%s' % (directory, (str(index + 1)).zfill(3), (str(index + 1)).zfill(3), '_seg.nii.gz')
        print(file)
        masks = nib.load(file)
        masks.set_data_dtype(np.int8)
        mask = masks.dataobj[:, :, 132]
        print(mask.shape)
        class_ids = []
        x, y, width, height = cv2.boundingRect(mask)

