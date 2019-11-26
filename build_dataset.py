"""
Usage:
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv
"""

# import packages
import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import random
import csv

# Construct argument parser
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True, help = "path to input images directory")
ap.add_argument("-l", "--labels", required=True, help = "path to annotations directory")
ap.add_argument("-r", "--train", required=True, help = "path to output training.csv file")
ap.add_argument("-e", "--test", required=True, help = "path to outpust test.csv file")
ap.add_argument("-c", "--classes", required=True, help = "path to output classes.csv file")
ap.add_argument("-s", "--split", default = 0.8, type = float, help = "train/test split")

args = vars(ap.parse_args())

# create var names for all arguments
annot_path = args["labels"]
images_path = args["input"]
train_csv = args["train"]
test_csv = args["test"]
classes_csv = args["classes"]
train_test_split = args["split"]

# grab image paths to construct train and test split on
image_paths = os.listdir(images_path)
training_length = int(len(image_paths) * train_test_split)
testing_length = int(len(image_paths) - training_length)
shuffled_set = random.sample(image_paths, len(image_paths))
train_img_paths = shuffled_set[0:training_length]
test_img_paths = shuffled_set[training_length:]

# define datasets to build (dType, list of paths, destination file)
dataset = [
("train", train_img_paths, train_csv),
("test", test_img_paths, test_csv)
]

# initialize set of classes and annotations
classes = set()
# annotations = []

# loop over train and test datasets
for (dType, imagePaths, outputCSV) in dataset:

    annotations = []
    # load the contents
    print("[INFO] creating '{}' set ...".format(dType))
    print("[INFO] {} total images in '{}'".format(len(imagePaths), dType))

    # open the output CSV file
    with open(outputCSV, "w") as o:
        writer = csv.writer(o)

        # loop over image paths
        for imagePath in imagePaths:
            # generate corresponding annotation path
            fname = os.path.basename(imagePath).split(".")[0]
            annotPath = os.path.join(annot_path, fname + ".xml")

            tree = ET.parse(annotPath)
            root = tree.getroot()

            # loop over all object elements
            for elem in root:
                if elem.tag == "object":
                    obj_name = None
                    coords = []
                    for subelem in elem:
                        # extract all label and bounding box coordinates
                        if subelem.tag == "name":
                            obj_name = subelem.text
                        if subelem.tag == "bndbox":
                            for subsubelem in subelem:
                                coords.append(int(float((subsubelem.text))))
                            xMin = coords[0]
                            yMin = coords[1]
                            xMax = coords[2]
                            yMax = coords[3]

                    # truncate any bounding box corrdinates that fall outside
                    # image boundaries
                    xMin = max(0, xMin)
                    yMin = max(0, yMin)
                    xMax = max(0, xMax)
                    yMax = max(0, yMax)

                    # ignore bounding boxes where minimum values are larger than
                    # max values and vice-versa (annotation errors)
                    if xMin >= xMax or yMin >= yMax:
                        continue
                    elif xMax <= xMin or yMax <= yMin:
                        continue

                    # gather image path, coordinates, and class for each object on image
                    item = [imagePath] + coords + [obj_name]
                    # create list of information of all objects in image
                    annotations.append(item)
                    # write to outputCSV
                    writer.writerows(annotations)
                    # update set of unique class labels
                    classes.add(obj_name)

    # write the annotations to file
    print("[INFO] writing {} annotations ...".format(dType))
    print("[INFO] total {} annotations".format(len(annotations)))
    with open(outputCSV, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(annotations)
    print("[INFO] {}.csv completed".format(dType))

# write the classes to file
print("[INFO] writing classes ...")
with open(classes_csv, 'w') as f:
    for i, line in enumerate(classes):
      f.write('{},{}\n'.format(line,i))
print("[INFO] classes.csv completed")

print("[FINAL] Task completed!")
