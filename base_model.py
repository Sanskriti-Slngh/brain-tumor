# import libraries
import os
from tensorflow import keras
import pickle
import random
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import numpy as np
import mrcnn
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize
import imgaug
import imgaug.augmenters as iaa
import nibabel as nib
import skimage
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

# get train and test datasets
from mrcnn_data_prep import train_set, val_set, train_set_t2, val_set_t2, test_set, test_set_t2

# random seed is 1234567890
random.seed(1234567890)

# base class for all models
# Describe each method (function)
class BaseModel:
    def __init__(self, params=None):
        self.params = {}
        # One line description for each variable
        self.params['resetHistory'] = False # parameter to reset the history of models, default = False
        self.params['models_dir'] = "." # model directory name used for locating and dumping items into files
        self.params['print_summary'] = True # prints model architecture as well as basic summary, default = True
        self.params['multi_classification'] = False # to perform multi-classification, default = False
        self.params['data_generation_enable'] = False # to generate more data, default = False
        self.params['force_normalize'] = False # to normalize the dataset, default = False
        self.params['scan_type'] = 't1' # this for two models, t1, t2
        self.params['augumentation'] = False
        self.params['overide'] = False
        self.params['coco'] = False

        self.patience = 4 # callback for training, default = 4
        self.is_train = True # default = True

        # setting parameters based on the params as given by the user
        if params is not None:
            for key, value in params.items():
                self.params[key] = value

        print("Scan type: " + self.params['scan_type'])

        # data
        if self.params['scan_type'] == 't1':
            self.train_set = train_set
            self.val_set = val_set
            self.test_set = val_set
        else:
            self.train_set = train_set_t2
            self.val_set = val_set_t2
            self.test_set = val_set_t2

        # including history class into BaseModel
        self.history = self.params['history']

        # combining model directory and name of model
        self.name = self.params['models_dir'] + '/mrcnn_model_no_change'

        # create model directory if unknown
        if not os.path.isdir(self.params['models_dir']):
            os.makedirs(self.params['models_dir'])

        # Copy the train.py
        if self.is_train:
            shutil.copyfile('train.py', self.params['models_dir'] + '/train.py.copy')

        print(self.name + '.h5py')
        print(os.path.isfile(self.name + '.h5py'))

        # Config class for model
        class BrainTumorConfig(Config):
            # Give the configuration a recognizable name
            NAME = "braintumor_detection"

            NUM_CLASSES = 1 + 1   # all images have brain tumors (this includes background)
            STEPS_PER_EPOCH = 50000
            LEARNING_RATE = 0.001
            IMAGE_MAX_DIM = 256
            IMAGE_SHAPE = [256,256,1]
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_NMS_THRESHOLD = 0.0

            DETECTION_MIN_CONFIDENCE = 0.9  # Skip detections with < 90% confidence

        self.config = BrainTumorConfig()
        self.config.display()

        # define the model
        self.model = modellib.MaskRCNN(mode='training', model_dir="C:/Users/Manish/projects/tiya/SF2021/models",
                                       config=self.config)
        #self.model2 = modellib.MaskRCNN(mode='inference', model_dir="C:/Users/Manish/projects/tiya/SF2021/models",
         #                              config=self.config)

        # load model if model is already there
        if not self.params['resetHistory'] and os.path.isfile(self.name + '.h5py') or self.params['overide']:
            print("Loading model from " + self.name + '.h5py')
            #self.model.load_weights(filepath=self.name + '.h5py',by_name=True)
            self.model.load_weights(filepath='C:/Users/Manish/projects/tiya/SF2021/models/experiments/t1-pneumo-aug/m_4.h5py', by_name=True)
            #self.model2.load_weights(filepath='C:/Users/Manish/projects/tiya/SF2021/models/experiments/t1_final/t1-aug/m_4.h5py',by_name=True)
            if params['predict']:
                with open('C:/Users/Manish/projects/tiya/SF2021/models/experiments/t1_final/m_3' + '.aux_data', 'rb') as fin:
                    self.history.loss, self.history.rpn_class_loss, self.history.rpn_bbox_loss, self.history.mrcnn_class_loss, self.history.mrcnn_bbox_loss, self.history.mrcnn_mask_loss, self.history.val_loss, self.history.val_rpn_class_loss, self.history.val_rpn_bbox_loss, self.history.val_mrcnn_class_loss, self.history.val_mrcnn_bbox_loss, self.history.val_mrcnn_mask_loss = pickle.load(fin)
        else:
            if self.params['coco']:
                self.model.load_weights(filepath='C:/Users/Manish/Downloads/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
            else:
                print('GOING THRO HERE')
                self.model.load_weights(filepath='C:/Users/Manish/Downloads/resent/final.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # save the model weights, biases, train_losses, val_losses, train_accuracy, val_accuracy
    def save(self, name=None):
        # replace name if specified
        self.sfname = self.name
        if name:
            self.sfname = name
        self.model.keras_model.save_weights(self.sfname + '.h5py')
        with open(self.sfname + '.aux_data', 'wb') as fout:
            pickle.dump((self.history.loss, self.history.rpn_class_loss, self.history.rpn_bbox_loss, self.history.mrcnn_class_loss, self.history.mrcnn_bbox_loss, self.history.mrcnn_mask_loss, self.history.val_loss, self.history.val_rpn_class_loss, self.history.val_rpn_bbox_loss, self.history.val_mrcnn_class_loss, self.history.val_mrcnn_bbox_loss, self.history.val_mrcnn_mask_loss), fout)
        if not name:
            plot_model(self.model, to_file=self.name + '.png')

    # mrcnn train model code
    def train(self):
        if self.params['augumentation']:
            print("DOING AUG")
            self.model.train(self.train_set, self.val_set, learning_rate=2 * self.config.LEARNING_RATE, epochs=5,
                             layers='heads', augmentation= imgaug.augmenters.Sometimes(1, [imgaug.augmenters.Fliplr(1), iaa.Rotate((-45, 45)), iaa.TranslateX(percent=(-0.1, 0.1)), iaa.TranslateY(percent=(-0.1, 0.1))]), user_callbacks=[self.history])
        else:
            # train weights (output layers or 'heads')
            self.model.train(self.train_set, self.val_set, learning_rate=2 * self.config.LEARNING_RATE, epochs=10, layers='heads', user_callbacks=[self.history])


    def predictSingle(self, img, id):
        print(self.config)
        # Load image
        imag = nib.load(img)
        idx = id%155
        image = imag.dataobj[:,:,idx]


        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        # return image
        # Run object detection

        results = self.model.detect([image], verbose=1)
        r = results[0]
        # show photo with bounding boxes, masks, class labels and scores
        mrcnn.visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['tumor'], r['scores'])

    # calculate the mAP for a model on a given dataset
    def evaluate_model(self):
        APs = list()
        ARs = list()
        for image_id in self.test_set.image_ids:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(self.test_set, self.config, image_id,
                                                                             use_mini_mask=False)
            scaled_image = mold_image(image, self.config)
            sample = np.expand_dims(scaled_image, 0)
            yhat = self.model.detect(sample, verbose=0)
           #yhat2 = self.model2.detect(sample, verbose=0)
            r = yhat[0]
            #r2 = yhat2[0]
            #print(gt_mask.size)
            if gt_mask.size:
                #print(gt_mask.size)
                #print(gt_mask)
                AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
                AR, _ = mrcnn.utils.compute_recall(r["rois"], gt_bbox, iou=0.5)
             #   AP2, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r2["rois"], r2["class_ids"], r2["scores"],r2['masks'])
              #  AR2, _ = mrcnn.utils.compute_recall(r["rois"], gt_bbox, iou=0.5)
                #AP = (AP*0.9+AP2*0.1)/2.0
                #AR = (AR*0.9+AR2*0.1)/2.0
            else:
                AR = 1.0
                AP = 1.0
            ARs.append(AR)
            APs.append(AP)

        # calculate the mean AP across all images
        mAP = np.mean(APs)
        mAR = np.mean(ARs)
        return mAP, mAR

    def train_plot(self, fig=None, ax=None, show_plot=True, label=None):
        if not fig:
            fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(self.history.mrcnn_bbox_loss[:], label=label + ' train', color='red')
        ax[0].plot(self.history.val_mrcnn_bbox_loss[:], label=label + ' val', color='blue')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epocs')
        ax[0].set_title("Loss vs epocs, train(Red)")

        ax[1].plot(self.history.mrcnn_mask_loss[:], label=label + ' train', color='red')
        ax[1].plot(self.history.val_mrcnn_mask_loss[:], label=label + ' val', color='blue')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('epocs')
        ax[1].set_title("Accuracy vs epocs, train(Red)")

        print('train loss: ' + str(self.history.loss[:]))
        print('train rpn class loss: ' + str(self.history.rpn_class_loss[:]))
        print('train rpn bounding box loss: ' + str(self.history.rpn_bbox_loss[:]))
        print('train class loss: ' + str(self.history.mrcnn_class_loss[:]))
        print('train bounding box loss: ' + str(self.history.mrcnn_bbox_loss[:]))
        print('train mask loss: ' + str(self.history.mrcnn_mask_loss[:]))
        print('epochs:   ' + str(len(self.history.mrcnn_mask_loss)))
        print('\n')
        print('val loss: ' + str(self.history.val_loss[:]))
        print('val rpn class loss: ' + str(self.history.val_rpn_class_loss[:]))
        print('val rpn bounding box loss: ' + str(self.history.val_rpn_bbox_loss[:]))
        print('val class loss: ' + str(self.history.val_mrcnn_class_loss[:]))
        print('val bounding box loss: ' + str(self.history.val_mrcnn_bbox_loss[:]))
        print('val mask loss: ' + str(self.history.val_mrcnn_mask_loss[:]))
        print('epochs:   ' + str(len(self.history.val_mrcnn_mask_loss)))

        if show_plot:
            plt.show()

    def predict(self):
        model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir='C:/Users/Manish/projects/tiya/SF2021/models')
        model.load_weights(filepath='C:/Users/Manish/projects/tiya/SF2021/models/experiments/current/m_3.h5py',
                           by_name=True)
        with open("C:/Users/Manish/projects/tiya/SF2021/data/valt1.dat", 'rb') as fin:
            x_val, y_val, auxillary_val = pickle.load(fin)
        # result = model.detect(x_val)
        images = []
        for image_id in range(8):
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.val_set, self.config, image_id+1,use_mini_mask=False)
            info = self.val_set.image_info[image_id+1]
            print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id+1, self.val_set.image_reference(image_id+1)))  # Run object detection
            images.append(image)
        results = model.detect(images, verbose=1)  # Display results
        r = results[0]
        for i in range(8):
            plt.imshow()
            visualize.display_instances(images[i], r['rois'], r['masks'], r['class_ids'],self.val_set.class_names, r['scores'], title="Predictions")
        return