import sys
import tensorflow.keras as keras
import nibabel as nib
import math
import cv2

# import models
from base_model import BaseModel

# LossHistory Class
class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.loss = []
        self.rpn_class_loss = []
        self.rpn_bbox_loss = []
        self.mrcnn_class_loss = []
        self.mrcnn_bbox_loss = []
        self.mrcnn_mask_loss = []
        self.val_loss = []
        self.val_rpn_class_loss = []
        self.val_rpn_bbox_loss = []
        self.val_mrcnn_class_loss = []
        self.val_mrcnn_bbox_loss = []
        self.val_mrcnn_mask_loss = []
        self.acc_epochs = 0
        super(LossHistory, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.rpn_class_loss.append(logs.get('rpn_class_loss'))
        self.rpn_bbox_loss.append(logs.get('rpn_bbox_loss'))
        self.mrcnn_class_loss.append(logs.get('mrcnn_class_loss'))
        self.mrcnn_bbox_loss.append(logs.get('mrcnn_bbox_loss'))
        self.mrcnn_mask_loss.append(logs.get('mrcnn_mask_loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_rpn_class_loss.append(logs.get('val_rpn_class_loss'))
        self.val_rpn_bbox_loss.append(logs.get('val_rpn_bbox_loss'))
        self.val_mrcnn_class_loss.append(logs.get('val_mrcnn_class_loss'))
        self.val_mrcnn_bbox_loss.append(logs.get('val_mrcnn_bbox_loss'))
        self.val_mrcnn_mask_loss.append(logs.get('val_mrcnn_mask_loss'))
        self.acc_epochs = self.acc_epochs + 1
        if epoch%1 == 0:
            # Save model
            print ("Saving the model in ./experiments/current/m_" + str(epoch))
            model.save(name='C:/Users/Manish/projects/tiya/SF2021/models/experiments/current/m_' + str(epoch))

params = {}
params["models_dir"] = 'C:/Users/Manish/projects/tiya/SF2021/models'
params['scan_type'] = 't2'
params['resetHistory'] = False
params['history'] = LossHistory()
params['predict'] = False
params['overide'] = False # 0.425
params['coco'] = False
params['augumentation'] = False


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
    return [[x_min, y_min, x_min + width, y_min + height]]

## Define the model
model = BaseModel(params)

if not params['predict']:
    model.train()
else:
    #test_mAP, mARs_test = model.evaluate_model()
    #print(test_mAP, mARs_test)

#    f_score_test = (2 * test_mAP * mARs_test) / (test_mAP + mARs_test)

 #   print('f1-score-test', f_score_test)
    model.predictSingle('C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_007/BraTS20_Training_007_t1.nii', 66)
    #print(extract_boxes('C:/Users/Manish/projects/tiya/SF2021/data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_002/BraTS20_Training_002_seg.nii', 29))
    #model.train_plot(label='none')
#    model.predict()
