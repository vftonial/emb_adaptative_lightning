################################################################################
######## ADAPTIVE LED MATRIX HIGH BEAM PROJECT #################################
################################################################################
# BY: CAN OZCIVELEK
# DATE: OCTOBER 2018
# DESCRIPTION: THIS PROJECT MAKES USE OF A CAMERA TO ANALYZE THE ROAD AND DETECT
#              VEHICLES. THEN  SENDS  THE  DETECTED  VEHICLES  POSITIONS TO  THE
#              CONTROLLER  WHICH  CONTROLS  THE  LED  MODULE TO TURN  INDIVIDUAL
#              PARTS ON/OFF SO  AS TO NOT DAZZLE OTHER DRIVERS. TO MAKE  VEHICLE
#              DETECTION POSSIBLE AT NIGHT, THE PROJECT MADE USE OF TENSORFLOW'S
#              OBJECT DETECTION API. CUSTOM  TRAINING HAS BEEN  DONE  USING OVER
#              500 IMAGES TO TRAIN  VIA TENSORFLOW'S ssd_inception_v2_coco MODEL
#
# THIS PROJECT CONSISTS OF 2 SYSTEMS IN COMMUNICATION:
# 1. PYTHON (VEHICLE DETECTION AT NIGHT)
# 2. ARDUINO (LED MODULE CONTROL)
################################################################################


# IMPORT NECESSARY LIBRARIES
import os
import cv2
import numpy as np
import tensorflow as tf
import time
#import pyautogui, sys
import serial
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

TTL = 50


################################################################################
######## START - MAIN FUNCTION #################################################
################################################################################

# Define the directory containing the object detection model we're using
MODEL_NAME = 'inference_graph'

# Get path to the current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, use_display_name = True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (ie data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# Initialize webcam feed
video = cv2.VideoCapture("damn_highbeams_edit.mp4")

img_on = cv2.imread("main-beam_cinza.png",cv2.IMREAD_COLOR)
img_off = cv2.imread("main-beam_cinza_cinza.png",cv2.IMREAD_COLOR)

cv2.namedWindow("Luz_farol")

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    height, width = frame.shape[:2]

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict = {image_tensor: frame_expanded})


    # Visualize the result
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.80)

    final_score = np.squeeze(scores)
    count = 0       # Variable to count the detected vehicles

    ############################################################################
    ### START - Loop to detect up to 5 vehicles per frame ####################
    for i in range(5):
        if scores is None or final_score[i] > 0.8:
            count = count + 1

    ### END - Loop to detect up to 5 vehicles per frame ######################
    ############################################################################

    #print(TTL)
    #print(count)
    
    if(count >= 1):
        #print("Cars detected. Reducing TTL.")
        TTL -= 1

    if TTL <= 0:
        cv2.imshow("Luz_farol", img_off)
        print("TTL less than 0. High Beams OFF.")

    # If zero vehicles detected, turn all the LEDs ON
    if(count == 0):
        #print("No cars detected. Increasing TTL.")
        TTL += 1

    if TTL > 0:
        cv2.imshow("Luz_farol", img_on)
        print("TTL higher than 0. High Beams ON.")

    if TTL >= 50:
        TTL = 50
    elif TTL <= -50:
        TTL = -50

    # Display the final image
    cv2.imshow('Vehicle Detection at Night', frame)

    # Press 'ENTER' to quit
    if cv2.waitKey(1) == 13:
        break
################################################################################
### END - Loop to play the input video #########################################

# Clean up
video.release()
cv2.destroyAllWindows()

################################################################################
######## END - MAIN FUNCTION ###################################################
################################################################################
