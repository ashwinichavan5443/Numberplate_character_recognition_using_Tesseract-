import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Watcher:
    DIRECTORY_TO_WATCH = "C:\Users\Krishna\Desktop\research1"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(0.0001)
        except:
            self.observer.stop()
            print (Error)

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        crowd= 0
        if event.is_directory:
            return None

        #elif event.event_type == 'created':
            # Take any action here when a file is first created.
            #print ("Received created event -" , event.src_path)

        elif event.event_type == 'created':
            # Taken any action here when a file is modified.
            #print ("Received modified event - " , event.src_path)
            crowd = 0
            cmd1=  " nohup ftp-upload -h 45.79.121.55 -u ftpuser --passive --password neon@123 -d abc/ " + event.src_path +" &"
            image_name = 'test18.jpg'+event.src_path[29:]
            image_name2 = event.src_path[29:]
	    imag = Image.open(image_name)
	    image = cv2.imread(image_name)

	    width, height = imag.size	
	    print(image_name2)
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


if __name__ == '__main__':
    w = Watcher()
    w.run()
