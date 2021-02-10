####################################################
#                                                  #
# ARDD Utils for Object Detection.                 #
# Created by Thomas Chia and Cindy Wu              #
# Medical Research by Sreya Devarakonda            #
# Created for the 2021 Congressional App Challenge #
# Winning "webapp" of Virginia's 11th District     #
#                                                  #
# Based on Joseph Redmon's                         #
#              YoloV3: An Incremental Improvement  #
#                                                  #
#################################################### 

import cv2
import time
import random
import colorsys
import PIL 
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from shutil import copyfile
from tqdm import tqdm
from .configs import *

def clahe_image(image_path, save_path = '', clipLimit = 1.0, channels = 'a'):
    """
    Applies Contrast Limited Histogram Equalization (CLAHE) on a given Image.

    Parameters:
        image_path (str): Path to the image.
        save_path (str): Location to where the image should be saved.
        clipLimit (float): The clip limit on the contrast in CLAHE.
        channels (r, g, b, a): Apply CLAHE to a specific channel or all channels.
    
    Returns:
        Image with CLAHE applied.

    Citation:
        CLAHE Preproccessing Function for Fundus Images
            Implementation: Thomas Chia, Cindy Wu: https://github.com/haoyuwu03/Intel/blob/master/image_preprocessing/clahe_g_channel.ipynb
    """

    # Step 1: Channel Splitting of the BGR Image.
    old_image = cv2.imread(image_path)
    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB) # Converts image to RGB
    R,G,B = cv2.split(old_image) # Splits the channels
    
    # Step 2: Apply ClAHE on the "g" channel of the image
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    
    if channels is 'g': 
        G = clahe.apply(G)
    if channels is 'b': 
        B = clahe.apply(B) 
    if channels is 'r': 
        R = clahe.apply(R) 
    if channels is 'a':
        G = clahe.apply(G)
        B = clahe.apply(B)
        R = clahe.apply(R)

    # Step 3: Merge Image Channels
    clahe_image = cv2.merge((R, G, B))

    return clahe_image

def detect_image(model, image_path, output_path, classes, input_size=416, show=False, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', colab = False, clahe = True):
    """
    Runs the Object Detection Neural Network on the input image, and returns a processed output.

    Parameters:
        model: The Tensorflow Model in the form of a .h5 model.
        image_path: Path to the input image.
        output_path: Path to save the output image.
        classes: Path to the classes list, which can be continuously updated as more lesions and conditions are updated.
        etc...
    
    Returns:
        Processed image and "image classes." Image classes contain the number of conditions detected per condition.
        
    """

    # Reads original image
    original_image      = cv2.imread(image_path)
    # Converts image to rgb format
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    if clahe == True:
        original_image = clahe_image(image_path, channels = 'a')

    # Preprocess the images so they are in the correct format
    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = tf.expand_dims(image_data, 0)

    # Predict the bounding box using the model
    pred_bbox = model.predict(image_data)
    # Reshape each prediction per bounding box
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    # Post process the image.
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold)
    image, symptom_classes = draw_bbox(original_image, bboxes, classes=classes, rectangle_colors=rectangle_colors)
    new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save the image
    if output_path != '': cv2.imwrite(output_path, new_image)

    # Display the image depending on requirements
    if (show == True) and (colab == True):
        plt.imshow(image)
        
    elif (show == True) and (colab == False):  
        cv2.imshow("predicted image", new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return image, symptom_classes

"""
The next following functions serve to process the images into a usable Tensor format. This includes converting the model outputs into usable coordinates,
preprocessing images into usable form factors, reading classnames, etc.
"""

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):   
    NUM_CLASS = read_class_names(classes)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    symptom_classes = np.zeros((27,), dtype=int)
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        symptom_classes[class_ind] += 1
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.5 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            score_str = " {:.2f}".format(score) if show_confidence else ""
            if tracking: score_str = " "+str(score)
            label = "{}".format(NUM_CLASS[class_ind]) + score_str
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image, symptom_classes

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)
            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)





