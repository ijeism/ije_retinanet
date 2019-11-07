"""
Image Inference:
Takes the path to the unseen inference images directory and outputs images with bounding
boxes around detections.
"""


# import modules
import os
import csv
import cv2
import glob
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


# construct argument parser
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input_dir", required=True, help = "dir path to inference images")
ap.add_argument("-t", "--threshold", default=0.5, type = float, help = "threshold for filtering weak detections")
ap.add_argument("-m", "--model", required=True, help = "path to trained/converted model")
ap.add_argument("-o", "--output_dir", required=True, help = "path to output directory")
ap.add_argument("-l", "--labels", required=True, help = "path to class csv")

args = vars(ap.parse_args())

# create variables
input_path = args["input_dir"]
output_path = args["output_dir"]
THRES_SCORE = args["threshold"]

#load class label mappings
LABELS = open(args["labels"]).read().strip().split('\n')
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

# load model
model = models.load_model(args["model"], backbone_name='resnet50')

# grab input image paths
inference_images = [os.path.join(input_path, file) for file in glob.glob(input_path + '*.jpg')]


# loop over inference images
for (i, img_path) in enumerate(inference_images):

    print("[INFO] predicting image {} of {}".format(i+1, len(inference_images)))

    # create file to store predictions in
    filename = os.path.basename(img_path).split('.')[0]
    output_file = os.path.join(output_path, filename + ".csv")

    #load image and preprocess
    image = read_image_bgr(img_path)
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    # detect objects in input image and correct for image scale
    start = time.time()
    (boxes, scores, labels) = model.predict_on_batch(image)
    print("processing time: ", time.time() - start)
    boxes /= scale

    items = []

    # loop over detections
    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):

        # weak detections are filtered
        if score < THRES_SCORE:
            continue
        # convert bounding box coordinates from float to int
        b = box.astype(int)

        # create a row for each valid detection
        item = [img_path, str(score), str(box[1]), str(box[0]), str(box[3]), str(box[2]), LABELS[label]]
        items.append(item)

        # write each row in the corresponding text file
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(items)

print("[FINAL] Predictions completed!")
