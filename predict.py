
""" Makes predictions on test/validation images using the trained model
and writes predictions to disk.
"""


# import packages
from keras_retinanet.utils.image import preprocess_image, read_image_bgr, resize_image
from keras_retinanet import models
from imutils import paths

import numpy as np
import argparse
import os

base_dir = os.getcwd()

# construct argument parser
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True, help = "path to pre-trained model")
ap.add_argument("-l", "--labels", default=base_dir + '/images_subset/classes.csv', help = "path to class labels")
ap.add_argument("-i", "--input", required=True, help = "path to directory containing input images")
ap.add_argument("-o", "--output", default=base_dir + '/data/', help = "path to directory to store predictions")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help = "minimum probability to filter weak detections")

args = vars(ap.parse_args())

# load class label mappings
LABELS = open(args['labels']).read().strip().split('\n')
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

# load the model and grab all input image paths
model = models.load_model(args["model"], backbone_name = "resnet50")
imagePaths = list(paths.list_images(args["input"]))

# loop over the input paths
for (i, imagePath) in enumerate(imagePaths):

    print("[INFO] predicting on image {} of {}".format(i+1, len(imagePaths)))

    #Create the filename to store the predictions in
    filename = (imagePath.split(os.path.sep)[-1]).split(".")[0]
    output_file = os.path.sep.join([args["output"], "{}.txt".format(filename)])
    file = open(output_file, 'w')

    # load input image, clone, and preprocess
    image = read_image_bgr(imagePath)
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis = 0)

    # detect object in the input image and correct for image scale
    (boxes, scores, labels) = model.predict_on_batch(image)
    boxes /= scale

    # loop over detections
    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        # filter out weak detections
        if score < args["confidence"]:
            continue

        # convert the bounding box coordinates from floats to integers
        box = box.astype("int")

        # create row for each prediction in the format:
        # <classname> <confidence> <ymin> <xmin> <ymax> <xmax>
        row = " ".join([LABELS[label], str(score), str(box[1]), str(box[0]), str(box[3]), str(box[2])])

        # write each row of prediction in the corresponding txt file
        file.write("{}\n".format(row))

    # close file
    file.close()
