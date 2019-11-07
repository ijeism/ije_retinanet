import os
import shutil
import urllib
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import csv
import pandas
import glob

# construct argument parser
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input_dir", required=True, help = "path to train images directory/")
ap.add_argument("-l", "--labels", required=True, help = "path to annotations directory/")
ap.add_argument("-a", "--annotation_csv", required=True, help = "path to annotations csv")
ap.add_argument("-c", "--classes_csv", required=True, help = "path to classes csv")

args = vars(ap.parse_args())


DATASET_DIR = args["input_dir"]
LABELS_DIR = args["labels"]
ANNOTATIONS_FILE = args["annotation_csv"]
CLASSES_FILE = args["classes_csv"]


annotations = []
classes = set([])

for xml_file in glob.glob(LABELS_DIR + "*.xml"):
    tree = ET.parse(os.path.join(DATASET_DIR, xml_file))
    root = tree.getroot()

    file_name = None

    for elem in root:
        if elem.tag == 'filename':
            file_name = os.path.join(DATASET_DIR, elem.text)

        if elem.tag == 'object':
            obj_name = None
            coords = []
            for subelem in elem:
                if subelem.tag == 'name':
                    obj_name = subelem.text
                if subelem.tag == 'bndbox':
                    for subsubelem in subelem:
                        coords.append(int(float((subsubelem.text))))
            item = [file_name] + coords + [obj_name]
            annotations.append(item)
            classes.add(obj_name)

with open(ANNOTATIONS_FILE, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(annotations)
print("[INFO] Annotations .csv completed")

with open(CLASSES_FILE, 'w') as f:
    for i, line in enumerate(classes):
      f.write('{},{}\n'.format(line,i))
print("[INFO] Classes .csv completed")
