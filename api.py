from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import io

import numpy as np

from PIL import Image

import shutil
import os



os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

app = FastAPI()

detect_fn = tf.saved_model.load('saved_model')
category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt",use_display_name=True)


def load_image_into_numpy_array(data):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(io.BytesIO(data)))


def predict(image):
     # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    image_np = load_image_into_numpy_array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    #print(input_tensor.shape)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)
    #print(detections)
    return Image.fromarray(image_np_with_detections)


@app.get("/")
async def main():
    return FileResponse(r"misdo.jpg")


@app.post("/uploadfile/")
async def create_upload_file(image: UploadFile = File(...)):
    img_data = await image.read()
    predicted_image = predict(img_data)
    #print(predicted_image.shape)
    predicted_image.save("Predicted.jpg")
    return "Success"