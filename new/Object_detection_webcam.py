
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pytesseract
import time
from PIL import Image
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'new_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
#PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
#PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap('birras_labelmap.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('trt.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

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
kernel = np.ones((5,5),np.uint8)
# Initialize webcam feed
#video = cv2.VideoCapture("rtsp://192.168.1.15/12")
video = cv2.VideoCapture(0)
#ret = video.set(3,640)
#ret = video.set(4,480)
count=0;
 #set frame height
#video.set(cv2.CAP_PROP_FPS,5)
while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT )
    start_time = time.time()
    # Perform the actual detection by running the model with the image as input
    boxes=0
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    
   
    	
    print( time.time()-start_time)
    ymin = boxes[0][0][0]*height
    xmin = boxes[0][0][1]*width 
    ymax = boxes[0][0][2]*height
    xmax = boxes[0][0][3]*width
    #print(ymin,ymax,xmin,xmax)
    frame2=frame    
    crop = frame2[int(ymin):int(ymax), int(xmin):int(xmax)]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    thresh=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image_name =str(count)+".jpg"
    cv2.imwrite(image_name,thresh)
    predicted_result = pytesseract.image_to_string(gray) 
    filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
    print(filter_predicted_result)
    count+=1
    vis_util.visualize_boxes_and_labels_on_image_array(
        	frame,
        	np.squeeze(boxes),
        	np.squeeze(classes).astype(np.int32),
        	np.squeeze(scores),
        	category_index,
        	use_normalized_coordinates=True,
        	line_thickness=8,
        	min_score_thresh=0.70)
    xmin=int(xmin)
    xmax=int(xmax)
    ymax=int(ymax)
    ymin=int(ymin)
    font=cv2.FONT_HERSHEY_DUPLEX
    #cv2.putText(frame,filter_predicted_result,(ymin-5,xmin-10),font,1,(0,0,255),1)
    cv2.imshow('filter_predicted_result', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

