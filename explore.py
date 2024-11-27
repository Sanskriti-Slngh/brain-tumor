import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np


def extract_boxes(file, image_id):
    # define anntation  file location
    path = file
    filename = nib.load(path)
    number = image_id % 155
    filename = filename.dataobj[:, :, number]
    x_min, y_min, width, height = cv2.boundingRect(filename)
    print(x_min, y_min, width, height)

    if math.isnan(x_min):
        print("working")
        exit()
    return x_min, y_min, x_min + width, y_min + height

x1, y1, width, height = extract_boxes("C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_002/BraTS20_Training_002_seg.nii", 50)
x = nib.load("C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_007/BraTS20_Training_007_seg.nii")
tumor_pixels = x.dataobj[:,:,50]

img2 = nib.load("C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_007/BraTS20_Training_007_t1.nii")
image2 = img2.dataobj[:,:,50]

img1 = nib.load("C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_007/BraTS20_Training_007_t1ce.nii")
image1 = img1.dataobj[:,:,50]
image = image1-image2

# Mark tumor pixels as red

cmap = plt.cm.gray
cmap.set_bad((1, 0, 0, 1))
h, w = tumor_pixels.shape
image3 = np.zeros((h,w))
for i in range(h):
    for j in range(w):
        if tumor_pixels[i,j]:
            print ("Marking pixels Nan")
            image3[i,j] = np.nan
        else:
            image3[i,j] = image[i,j]

plt.imshow(image3, cmap='gray')
plt.show()