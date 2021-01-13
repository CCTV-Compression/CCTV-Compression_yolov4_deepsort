from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train, filter_boxes
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)
#모델 생성
input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = np.array(cfg.YOLO.STRIDES), get_anchors(cfg.YOLO.ANCHORS, False), len(read_class_names(cfg.YOLO.CLASSES)), cfg.YOLO.XYSCALE
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

freeze_layers = utils.load_freeze_layer('yolov4', False)

feature_maps = YOLO(input_layer, NUM_CLASS, 'yolov4', False)
if False:
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        else:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(fm)
        bbox_tensors.append(bbox_tensor)
else:
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        elif i == 1:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        else:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(fm)
        bbox_tensors.append(bbox_tensor)
model = tf.keras.Model(input_layer, bbox_tensors)
#utils.load_weights(model, '/home/ubuntu/YOLOv4/checkpoints/admm_pruning/retraining/yolov4.data-00000-of-00001', 'yolov4', False)
model.load_weights('./checkpoints/yolov4')
layers = model.layers

zero_count = 0
size = 0
for layer in layers:
    if 'conv2d' in layer.name and layer.name not in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
        zero_count += np.sum(layer.get_weights()[0]==0)
        size += layer.get_weights()[0].size
print(zero_count)
print(size)
print(zero_count/size)



