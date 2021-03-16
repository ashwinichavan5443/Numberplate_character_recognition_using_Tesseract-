import os
import cv2
import numpy as np
import pandas as data
import sys
import pytesseract # this is tesseract module 
from pytesseract import Output
import matplotlib.pyplot as plt 
import glob
from PIL import Image
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
# construct the argument parse and parse the arguments


IMAGE_NAME = 'test (68).jpg'
#imag = imageio.imread(IMAGE_NAME,as_gray=True)
image = cv2.imread(IMAGE_NAME)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fm = variance_of_laplacian(gray)
text = "Not Blurry"
# if the focus measure is less than the supplied threshold,
# then the image should be considered "blurry"
if fm < 100.0:
	text = "Blurry"
if fm < 3800.0:
	text = "Blurry"
# show the image
print(text,fm)
#(thresh, blackAndWhiteImage) = cv2.threshold(gray, 112, 255, cv2.THRESH_BINARY)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
	angle = -(90 + angle)
# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle
(h, w) =thresh.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(thresh, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
#	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("rotate",rotated)
rot=rotated.copy()
boxes = pytesseract.image_to_boxes(rotated)
h, w = rotated.shape
custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
crp=0
x=[]
count=0
img=0
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(rotated, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 255), 2)
    
print(boxes)

cv2.imshow('test',img)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3,3), 0) 
kernel = np.ones((3,3), np.uint8)
#sharpen_kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img_erosion = cv2.erode(gray, sharpen_kernel, iterations=1)
dilation = cv2.dilate(img_erosion,sharpen_kernel,iterations = 1)
sharpen = cv2.filter2D(img_erosion, -1, sharpen_kernel)
test=cv2.bilateralFilter(dilation, -1, -15, -15);
#cv2.imshow('test',image)


predicted_result = pytesseract.image_to_string(rot,lang='eng',config=custom_config )
filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
text = pytesseract.image_to_data(rotated, output_type='data.frame')
text = text[text.conf != -1]
conf = text.groupby(['block_num'])['conf'].mean()

print(filter_predicted_result,conf)
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()