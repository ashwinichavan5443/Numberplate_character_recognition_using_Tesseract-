import os
import time
import cv2
import pytesseract
import numpy as np

img =cv2.imread("1610519195.8773115.jpg")
gray_imag = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoise = cv2.GaussianBlur(gray_imag, (3, 3), 0)
thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#cv2.imshow("Image", img)
#cv2.waitKey(0)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

for c in contours:
#	M = cv2.moments(c)
#	print(M)
#	cX = int(M["m10"] / M["m00"])
#	cY = int(M["m01"] / M["m00"])
#	print(cX,cY)
	area = cv2.contourArea(c)
	#print(area)
	if area > 8000:
		x,y,w,h = cv2.boundingRect(c)
		print(x,y,w,h)
		ROI = opening[y:y+h, x:x+w]
#	cv2.drawContours(thresh, [c], -1, (0, 255, 0), 2)
	#cv2.circle(thresh, (cX, cY), 7, (255, 255, 255), -1)
	#cv2.putText(thresh, "center", (cX - 20, cY - 20),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	cv2.imshow("Image", opening)
	cv2.waitKey(0)
cv2.imwrite('ROI.png',ROI)
print(pytesseract.image_to_string(cv2.imread("ROI.png"),lang="eng", config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMOPQRSTUVWXYZ0123456789'))


#cv2.imwrite("/home/cloudvms/ashwini/gray.png",thresh)
