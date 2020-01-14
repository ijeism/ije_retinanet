# ije_retinanet

This repository is designed to be used with [`keras-retinanet`](https://github.com/fizyr/keras-retinanet) - Fizyr's popular Keras implementation of the RetinaNet object detection. 

It can be used to:
* Prepare your dataset (both training and test sets) in the CSV format required for training and evaluation with `keras-retinanet` (`build_dataset.py`)
* Generate predictions (class and bounding box coordinates) on new images and output them to CSV files (`image_inference_write.py`)
* Generate and save copies of original test/inference images with bounding boxes drawn around detected objects (`image_inference_print.py`)

[This Jupyter Notebook](https://colab.research.google.com/drive/1RVG7MqoGnIku0oXPSMjmT_T-o1aU9y_O#scrollTo=ryr56it7WavU) and [this post](https://medium.com/@ije_good/object-detection-on-satellite-imagery-using-retinanet-part-1-training-e589975afbd5) provide examples of how this repo can be useful.
