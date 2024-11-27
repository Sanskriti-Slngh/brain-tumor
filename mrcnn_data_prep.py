import nibabel as nib
import cv2
import math
import numpy as np
import mrcnn.utils
import pickle

with open("C:/Users/Manish/projects/tiya/SF2021/data/file_names.dat", 'rb') as fin:
    t1_train_files, t1_train_files_y, t1_val_files, t1_val_files_y, t1_test_files, t1_test_files_y, t2_train_files, t2_train_files_y, t2_val_files, t2_val_files_y, t2_test_files, t2_test_files_y = pickle.load(fin)

main_dir = 'C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/'

class BrainTumor(mrcnn.utils.Dataset):
    # load the dataset definitions
    def load_dataset(self, is_train=False, is_val=False):

        # Add classes. We have only one class to add.
        self.is_train = is_train
        self.is_val = is_val
        self.add_class("dataset", 1, "brain tumor")
        image_id = 0
        if is_train:
            for i in range(t1_train_files.shape[0]):
                image_id = image_id + 1
                img_path = main_dir + t1_train_files[i-1]
                ann_path = main_dir + t1_train_files_y[i-1]
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
        elif is_val:
            for i in range(t1_val_files.shape[0]):
                image_id = image_id + 1
                img_path = main_dir + t1_val_files[i-1]
                ann_path = main_dir + t1_val_files_y[i-1]
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
        else:
            for i in range(t1_test_files.shape[0]):
                image_id = image_id + 1
                img_path = main_dir + t1_test_files[i-1]
                ann_path = main_dir + t1_test_files_y[i-1]
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    # UNUSED
    def extract_boxes(self, image_id):
        print("hi")
        # get details of image
        info = self.image_info[image_id]
        # define anntation  file location
        path = info['annotation']
        filename = nib.load(path)
        number = image_id % 155
        filename = filename.dataobj[:, :, number]
        x_min, y_min, width, height = cv2.boundingRect(filename)
        print(x_min, y_min, width, height)

        if math.isnan(x_min):
            print("working")
            exit()
        return [[x_min, y_min, x_min + width, y_min + height]]

    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define anntation  file location
        path = info['annotation']
        masks = nib.load(path)
        number = image_id % 155
        masks = masks.dataobj[:, :, number]
        mask = np.reshape(masks, (240, 240, 1))
        class_ids = []
        if np.count_nonzero(mask) < 201:
            mask = np.zeros((240, 240, 1))
        class_ids.append(self.class_names.index('brain tumor'))
        return mask, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']

class BrainTumort2(mrcnn.utils.Dataset):
    # load the dataset definitions
    def load_dataset(self, is_train=False, is_val=False):

        # Add classes. We have only one class to add.
        self.is_train = is_train
        self.is_val = is_val
        self.add_class("dataset", 1, "brain tumor")
        image_id = 0
        if is_train:
            for i in range(t2_train_files.shape[0]):
                image_id = image_id + 1
                img_path = main_dir + t2_train_files[i-1]
                ann_path = main_dir + t2_train_files_y[i-1]
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
        elif is_val:
            for i in range(t2_val_files.shape[0]):
                image_id = image_id + 1
                img_path = main_dir + t2_val_files[i-1]
                ann_path = main_dir + t2_val_files_y[i-1]
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
        else:
            for i in range(t2_test_files.shape[0]):
                image_id = image_id + 1
                img_path = main_dir + t2_test_files[i-1]
                ann_path = main_dir + t2_test_files_y[i-1]
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    # UNUSED
    def extract_boxes(self, image_id):
        print("hi")
        # get details of image
        info = self.image_info[image_id]
        # define anntation  file location
        path = info['annotation']
        filename = nib.load(path)
        number = image_id % 155
        filename = filename.dataobj[:, :, number]
        x_min, y_min, width, height = cv2.boundingRect(filename)
        print(x_min, y_min, width, height)

        if math.isnan(x_min):
            print("working")
            exit()
        return [[x_min, y_min, x_min + width, y_min + height]]

    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define anntation  file location
        path = info['annotation']

        masks = nib.load(path)
        number = image_id % 155
        masks = masks.dataobj[:, :, number]
        masks = masks.astype(np.int8)
        mask = np.reshape(masks, (240, 240, 1))
        if np.count_nonzero(mask) < 201:
            mask = np.zeros((240, 240, 1))
        class_ids = []
        class_ids.append(self.class_names.index('brain tumor'))
        return mask, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']

# T1
# prepare train set
train_set = BrainTumor()
train_set.load_dataset(is_train=True, is_val=False)
train_set.prepare()
print("Train: %d" % len(train_set.image_ids))
# prepare val set
val_set = BrainTumor()
val_set.load_dataset(is_train=False, is_val=True)
val_set.prepare()
print("Val: %d" % len(val_set.image_ids))
# prepare test set
test_set = BrainTumor()
test_set.load_dataset(is_train=False, is_val=False)
test_set.prepare()
print("Test: %d" % len(test_set.image_ids))


# T2
# prepare train set
train_set_t2 = BrainTumort2()
train_set_t2.load_dataset(is_train=True, is_val=False)
train_set_t2.prepare()
print("Train: %d" % len(train_set_t2.image_ids))
# prepare val set
val_set_t2 = BrainTumort2()
val_set_t2.load_dataset(is_train=False, is_val=True)
val_set_t2.prepare()
print("Val: %d" % len(val_set_t2.image_ids))
# prepare test set
test_set_t2 = BrainTumort2()
test_set_t2.load_dataset(is_train=False, is_val=False)
test_set_t2.prepare()
print("Test: %d" % len(test_set_t2.image_ids))

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