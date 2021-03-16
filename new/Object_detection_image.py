# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pytesseract # this is tesseract module 
import matplotlib.pyplot as plt 
import glob
from PIL import Image
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
#our model is 'frozen_inference_graph'
#PATH_TO_CKPT = os.path.join('new_graph','frozen_inference_graph.pb')

# Path to label map file
#PATH_TO_LABELS = os.path.join('training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

#load labelmap
label_map = label_map_util.load_labelmap('labelmap.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# Load the Tensorflow model into memory.
print("model loading");
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def =   tf.GraphDef()
    with tf.io.gfile.GFile('trt.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)
print("model loaded");
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
IMAGE_NAME = 'test18.jpg'
imag = Image.open(IMAGE_NAME)
image = cv2.imread(IMAGE_NAME)

width, height = imag.size    
count=1
# Perform the actual detection by running the model with the image as input
list_license_plates = [] 
predicted_license_plates = []
index=0
while(count>=1):
	
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_expanded = np.expand_dims(image_rgb, axis=0)
	(boxes, scores, classes, num) = sess.run(
     			 [detection_boxes, detection_scores, detection_classes, num_detections],
      			feed_dict={image_tensor: image_expanded})
	classes1=np.squeeze(scores).tolist()
	while classes1[index]>0.60:
		print(classes1[index])
		
		print("object detected")
		#print(scores)
	
        
		ymin = boxes[0][index][0]*height
		xmin = boxes[0][index][1]*width 
		ymax = boxes[0][index][2]*height
		xmax = boxes[0][index][3]*width
		print('ymin=',ymin,'ymax=',ymax,'xmin=',xmin)
		
		crop = image[int(ymin):int(ymax), int(xmin):int(xmax)]
		#result = cv2.GaussianBlur(crop, (5,5), 0)
		#thresh = cv2.threshold(crop, 100, 255, cv2.THRESH_BINARY_INV)[1]
		#result = cv2.GaussianBlur(thresh, (5,5), 0)
		#result = 255 - result
		gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
		ret, thresh1 = cv2.threshold(gray, 100, 230,  cv2.THRESH_OTSU )
		#threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		#result = cv2.GaussianBlur(thresh, (3,3), 0)
		#
		#dilation = cv2.dilate(crop,kernel,iterations = 1)
		thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
		#result = cv2.GaussianBlur(thresh1, (3,3), 0)
		#gray = cv2.bilateralFilter(thresh, 11, 17, 17)
		#result2=cv2.filter2D(thresh1,-1,kernel)	
		cv2.imshow('result2',thresh1)
		custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
		
		#boxe = pytesseract.image_to_boxes(threshold_img)
		
		#h, w= threshold_img.shape
		#for b in boxe.splitlines():
		#	b = b.split(' ')
		#	thresh1 = cv2.rectangle(threshold_img, (int(b[1]),h - int(b[2])), (int(b[3]),h - int(b[4])), (0, 255, 0), 2)
		#	cv2.imshow('result2',threshold_img)
		#print(boxe)
		predicted_result = pytesseract.image_to_string(thresh1,lang='eng',config=custom_config)
 
		
		image_name ="RJImage"+".jpg"
		cv2.imwrite(image_name,gray)
		filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
		print(filter_predicted_result)
                	
		xmin=int(xmin)
		xmax=int(xmax)
		ymax=int(ymax)
		ymin=int(ymin)
		cv2.rectangle(image,(xmin-5,ymin-35),(xmin+120,ymin-60),(0,0,255),-1)
		font=cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image,filter_predicted_result,(xmin-5,ymin-40),font,0.5,(255,255,255),1)
		index+=1
	vis_util.visualize_boxes_and_labels_on_image_array(
    		image,
    		np.squeeze(boxes),
    		np.squeeze(classes).astype(np.int32),
    		np.squeeze(scores),
    		category_index,
  	        use_normalized_coordinates=True,
    		line_thickness=6,
   	 	min_score_thresh=0.60)
	
        
	
	
	cv2.imshow('filter_predicted_result', image)
	count=count-1

# All the results have been drawn on image. Now display the 
# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
