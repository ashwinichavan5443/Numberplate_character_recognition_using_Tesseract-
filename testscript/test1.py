import cv2
import numpy as np
import pytesseract
image = cv2.imread('test (5).jpg')
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem 3 --psm 10'
# Filter for ROI using contour area and aspect ratio
cnts= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
count=10

cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for  c in cnts:
    
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    if area > 2000 and aspect_ratio > .5:
        mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        data = pytesseract.image_to_string(mask, lang='eng', config=custom_config)
        print(data)
boxes = pytesseract.image_to_boxes(mask)
h, w,_ = mask.shape
custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
crp=0
x=[]
count=0
img=0
for b in boxes.splitlines():
    b = b.split(' ')
    #img = cv2.rectangle(mask, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 255), 2)
    
    
print(boxes)

cv2.imshow('test',img)

        
cv2.imshow('mask',mask)
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
rot = cv2.warpAffine(thresh, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
#	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("rotate",rot)
# Perfrom OCR with Pytesseract
data = pytesseract.image_to_string(thresh, lang='eng', config=custom_config)
print(data)

cv2.imshow('thresh', thresh)
cv2.imshow('mask', mask)
cv2.waitKey()

