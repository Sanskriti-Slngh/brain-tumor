# import libraries
import nibabel as nib
import numpy as np
import pickle
import mrcnn.utils
import cv2
import random
import os
import math

# random seed 10000
np.random.seed(10000)

# main directory where data is stored
main_dir = 'C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/'

# file end names are created for data sets t1 and t2
file_end_name_t1 = ['_t1.nii.gz', '_t1ce.nii.gz']
file_end_name_t2 = ['_flair.nii.gz', '_t2.nii.gz']
file_end = [file_end_name_t1, file_end_name_t2]
t = ['t1','t2']

# additional variables that come into use during the making of classes
t1_train_files = []
t1_val_files = []
t1_test_files = []
t1_train_files_y = []
t1_val_files_y = []
t1_test_files_y = []
t2_train_files = []
t2_val_files = []
t2_test_files = []
t2_train_files_y = []
t2_val_files_y = []
t2_test_files_y = []

# True - goes through process of creating data, False - overides and opens data
overide = True

if overide:
    for q in range(2):
        all_files_x = []
        all_files_y = []

        # list containing the end of nii file names
        file_end_name = file_end[q]

        # step 1: get x(pixel data) and y(mask pixel values) and auxillary(3d image picture, section from picture)
        data_total_x = []
        data_total_y = []
        data_total_auxillary = []

        # variable creation
        img_idx = 0

        # there are 369 different patient folders with 5 different files in each
        # the shape of each image is (240, 240, 155)
        # 5 files: t1-scan (x), t1-enhanced scan (x), t2-scan (x), t2-enhanced-flair scan (x), segmentation mask (y)
        for index in range(369):
            img_idx = img_idx + 1
            for j in range(155):
                for i in range(2):

                    all_files_x.append('BraTS20_Training_%s/BraTS20_Training_%s%s'%((str(index+1)).zfill(3),(str(index+1)).zfill(3),file_end_name[i]))
                    img = nib.load('%sBraTS20_Training_%s/BraTS20_Training_%s%s'%(main_dir,(str(index+1)).zfill(3),(str(index+1)).zfill(3),file_end_name[i]))
                    fdata = img.dataobj[:,:,j]
                    fdata = fdata.astype(np.int8)
                    data_total_x.append(fdata)

                    all_files_y.append('BraTS20_Training_%s/BraTS20_Training_%s%s'%((str(index+1)).zfill(3),(str(index+1)).zfill(3),'_seg.nii.gz'))
                    img = nib.load('%sBraTS20_Training_%s/BraTS20_Training_%s%s'%(main_dir,(str(index+1)).zfill(3),(str(index+1)).zfill(3),'_seg.nii.gz'))
                    fdata = img.dataobj[:,:,j]
                    fdata = fdata.astype(np.int8)
                    data_total_y.append(fdata)
                    data_total_auxillary.append((img_idx, j))

        # turn lists into numpy arrays
        data_total_x = np.array(data_total_x)
        data_total_y = np.array(data_total_y)
        data_total_auxillary = np.array(data_total_auxillary)
        all_files_x = np.array(all_files_x)
        all_files_y = np.array(all_files_y)

        # print shape of x and y before reshape
        print(data_total_x.shape)
        print(data_total_y.shape)
        print(data_total_auxillary.shape)
        print(all_files_x.shape)
        print(all_files_y.shape)

        # variable needed for reshape
        x_image = data_total_x.shape[0]
        y_image = data_total_y.shape[0]

        assert(x_image == y_image), "ERROR"
        if x_image != y_image:
            print("ERROR")
            exit()

        # reshape both into correct sizes
        data_total_x = np.reshape(data_total_x, (240, 240, x_image))
        data_total_y = np.reshape(data_total_y, (240, 240, y_image))

        # print shape of x and y after reshape
        print(data_total_x.shape)
        print(data_total_y.shape)

        # delete earlier variable
        del x_image
        del y_image

        # train/val/test split
        # test = 10% total
        n = data_total_x.shape[-1]
        indices = np.random.choice(n, n, replace=False)
        aaa = int(n*0.9)

        x_train_all = data_total_x[:,:,indices[0:aaa]]
        y_train_all = data_total_y[:,:,indices[0:aaa]]
        auxillary_train_all = data_total_auxillary[indices[0:aaa]]

        x_test = data_total_x[:,:,indices[aaa:]]
        y_test = data_total_y[:,:,indices[aaa:]]
        auxillary_test = data_total_auxillary[indices[aaa:]]

        if q == 0:
            t1_train_files_all = all_files_x[indices[0:aaa]]
            t1_train_files_y_all = all_files_y[indices[0:aaa]]
            t1_test_files = all_files_x[indices[aaa:]]
            t1_test_files_y = all_files_y[indices[aaa:]]
        else:
            t2_train_files_all = all_files_x[indices[0:aaa]]
            t2_train_files_y_all = all_files_y[indices[0:aaa]]
            t2_test_files = all_files_x[indices[aaa:]]
            t2_test_files_y = all_files_y[indices[aaa:]]

        # train/val split
        # val = 10% total train
        n = x_train_all.shape[-1]
        indices = np.random.choice(n, n, replace=False)
        aaa = int(n*0.9)

        x_train = x_train_all[:,:,indices[0:aaa]]
        y_train = y_train_all[:,:,indices[0:aaa]]
        auxillary_train = data_total_auxillary[indices[0:aaa]]

        x_val = x_train_all[:,:,indices[aaa:]]
        y_val = y_train_all[:,:,indices[aaa:]]
        auxillary_val = data_total_auxillary[indices[aaa:]]

        if q == 0:
            t1_train_files = t1_train_files_all[indices[0:aaa]]
            t1_train_files_y = t1_train_files_y_all[indices[0:aaa]]
            t1_val_files = t1_train_files_all[indices[aaa:]]
            t1_val_files_y = t1_train_files_y_all[indices[aaa:]]
        else:
            t2_train_files = t2_train_files_all[indices[0:aaa]]
            t2_train_files_y = t2_train_files_y_all[indices[0:aaa]]
            t2_val_files = t2_train_files_all[indices[aaa:]]
            t2_val_files_y = t2_train_files_y_all[indices[aaa:]]

        # turning every list into a numpy array
        t1_train_files = np.array(t1_train_files)
        t1_train_files_y = np.array(t1_train_files_y)
        t1_val_files = np.array(t1_val_files)
        t1_val_files_y = np.array(t1_val_files_y)
        t1_test_files = np.array(t1_test_files)
        t1_test_files_y = np.array(t1_test_files_y)

        t2_train_files = np.array(t2_train_files)
        t2_train_files_y = np.array(t2_train_files_y)
        t2_val_files = np.array(t2_val_files)
        t2_val_files_y = np.array(t2_val_files_y)
        t2_test_files = np.array(t2_test_files)
        t2_test_files_y = np.array(t2_test_files_y)

        # print every data shape
        print(x_train.shape)
        print(y_train.shape)
        print(auxillary_train.shape)
        print(x_val.shape)
        print(y_val.shape)
        print(auxillary_val.shape)
        print(x_test.shape)
        print(y_test.shape)
        print(auxillary_test.shape)

        # save train, val, test data in file
        with open("C:/Users/Manish/projects/tiya/SF2021/data/train" + t[q] + ".dat", 'wb') as fout:
            pickle.dump((x_train, y_train, auxillary_train), fout, protocol=4)

        with open("C:/Users/Manish/projects/tiya/SF2021/data/val" + t[q] + ".dat", 'wb') as fout:
            pickle.dump((x_val, y_val, auxillary_val), fout, protocol=4)

        with open("C:/Users/Manish/projects/tiya/SF2021/data/test" + t[q] + ".dat", 'wb') as fout:
            pickle.dump((x_test, y_test, auxillary_test), fout, protocol=4)

    with open("C:/Users/Manish/projects/tiya/SF2021/data/file_names.dat",'wb') as fout:
        pickle.dump((t1_train_files, t1_train_files_y, t1_val_files, t1_val_files_y, t1_test_files, t1_test_files_y, t2_train_files, t2_train_files_y, t2_val_files, t2_val_files_y, t2_test_files, t2_test_files_y), fout, protocol=4)

# delete excess variables
del t1_train_files
del t1_val_files
del t1_test_files
del t1_train_files_y
del t1_val_files_y
del t1_test_files_y
del t2_train_files
del t2_val_files
del t2_test_files
del t2_train_files_y
del t2_val_files_y
del t2_test_files_y
del file_end_name_t2
del file_end_name_t1
# del x_train_all
# del y_train_all
# del all_files_x
# del all_files_y
# del x_test
# del x_val
# del x_train
# del y_test
# del y_val
# del y_train
# del auxillary_train_all
# del auxillary_val
# del auxillary_train
# del auxillary_test
# del data_total_auxillary

