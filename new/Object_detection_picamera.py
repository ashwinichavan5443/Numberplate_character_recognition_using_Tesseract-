#---------- Deteccion de objetos con PiCamera usando Tensorflow ----------
#
# Luis Arias Gomez, Edward Umana Williams, Guillermo Lopez Navarro
# Reconocimiento de Patrones - Sistemas embebidos de Alto Desempeno
# Tecnologico de Costa Rica
# Agosto 2019

# Este programa utiliza un clasificador con TensorFlow para deteccion de objetos.
# Se carga un modelo entrenado al cual se le brinda imagenes capturadas con la 
#    camara, lo cual retorna una etiqueta que posteriormente se utiliza en conjunto
#    con un marco que rodea el objeto, para denotar su identidad segun lo predijo
#    el modelo.

# Este codigo se basa parcialmente en el ejemplo de Google disponible en
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# Asimismo, se utilizo la guia disponible en el link a continuacion, para la 
#     integracion de la camara RPI con la Jetson Nano.
# https://www.jetsonhacks.com/2019/04/02/jetson-nano-raspberry-pi-camera/

import os
import cv2
import time
import numpy as np
import tensorflow as tf
import pytesseract
# Utilities para etiquetado en el video a mostrar y para manipulacion de
#     las etiquetas del modelo, respectivamente
from utils import visualization_utils as vis_util
from utils import label_map_util
from threading import Thread, Lock
# Dimensiones del video a mostrar
IM_WIDTH = 640#720 #1280 # 640
IM_HEIGHT = 480#540 #720 # 480

# Funcion que retorna handler de la camara RPI
#def gstreamer_pipeline (capture_width=IM_WIDTH, capture_height=IM_HEIGHT, display_width=IM_WIDTH, display_height=IM_HEIGHT, framerate=60, flip_method=0) :   
#    return ('nvarguscamerasrc ! ' 
#    'video/x-raw(memory:NVMM), '
#    'width=(int)%d, height=(int)%d, '
#    'format=(string)NV12, framerate=(fraction)%d/1 ! '
#    'nvvidconv flip-method=%d ! '
#    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
#    'videoconvert ! '
#    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

class WebcamVideoStream :
    def __init__(self, src) :
        self.stream = False
        self.sr = src
        print("Init Called :") 
        #print("buffer ",self.stream.get(cv2.CAP_PROP_BUFFERSIZE))
        #self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        #self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        #(self.grabbed, self.frame1) = self.stream.read()
        self.started = False
        self.released = False
        self.read_lock = Lock()

    def start(self) :
        self.stream = cv2.VideoCapture(self.sr)
        if self.started :
            print ("already started!!")
            #return None
        self.started = True
        self.t1 = Thread(target=self.update, args=())
        self.t1.start()
        return self

    def update(self) :
        while True :
            (grabbed, frame1) = self.stream.read()
            #self.read_lock.acquire()
            if grabbed:
                self.grabbed, self.frame1 = grabbed, frame1
            else:
                print("Frame NOT Captured")
                self.stream.release()
                self.released = True
                self.grabbed = None
                time.sleep(5)
                #self.t1.join(timeout=1)
    def read(self) :
        #self.read_lock.acquire()
        if self.released :
            print("Reconnecting Stream..")
            self.released = False
            self.stream = cv2.VideoCapture(self.sr,cv2.CAP_FFMPEG)
            time.sleep(1)
        if self and self.grabbed:
            self.read_lock.acquire()
            frame1 = self.frame1.copy()
            self.read_lock.release()
            #print("Frame Captured")
            return frame1
        else:
            #self.t1.join(timeout=1)
            #self.stream.release()
            #time.sleep(0.01) 
            return None
    def stop(self) :
        self.started = False
        #time.sleep(1) thread.kill() added and sleep(1) added for delay for staring new source
        self.stream.release()
        #self.t1.join()
        #self.thread.kill()
        time.sleep(0.01)

    def __exit__(self, exc_type, exc_value, traceback) :
        self.thread.join()
        time.sleep(0.01) 
        self.stream.release()


# Nombre del directorio que contiene el modelo a utilizar para la prediccion
#MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_NAME = 'birras'
#LABELS = 'mscoco_label_map.pbtxt'
#LABELS = 'birras_labelmap_6.pbtxt'
LABELS = 'birras_labelmap.pbtxt'

# Numero de clases que puede identificar el modelo
#NUM_CLASSES = 90
#NUM_CLASSES = 6
NUM_CLASSES = 1

# Directorio actual
CWD_PATH = os.getcwd()

# Ruta al archivo .pb (modelo a utilizar)
#PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
#PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph_LOSS_1,5.pb')

# Ruta al archivo que contiene etiquetas mapeadas a identificadores de objeto
# Este mapeo permite identificar con un nombre legible el valor predicho por 
#     la red convolutiva
PATH_TO_LABELS = os.path.join(CWD_PATH, 'labels', LABELS)


# Cargamos el mapeo de etiquetas.
# Para ello recurrimos a una libreria especializada, cargada previamente
label_map = label_map_util.load_labelmap('birras_labelmap.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Cargamos el modelo de TensorFlow en memoria
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    TFSess = tf.compat.v1.Session(graph=detection_graph)

# Definimos los tensores de entrada y salida para el clasificador
# El tensor de entrada es cada cuadro del video (una imagen)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Los tensores de salida corresponden a las cajas de deteccion, scores, y clases
# Las cajas corresponden a la parte de la imagen que contiene un objeto detectado
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Cada score representa el porcentaje de asertividad de la prediccion
# El score se muestra en conjunto con la etiqueta asignada al objeto detectado
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Numero de objetos detectados
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Inicializar calculo de FPS (cuadros por segundo), para mostrarlo en pantalla
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Nombre de la ventana del video a mostrar
WIN_NAME = 'Numberplate Detection'

# Inicializamos la camara
#cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture("rtsp://user1:user12345@192.168.1.13:554/h264/ch3/sub/av_stream",cv2.CAP_FFMPEG)
#cap = WebcamVideoStream("rtsp://192.168.1.15/12").start()
cap = cv2.VideoCapture("rtsp://192.168.1.15/12")
#if cap.isOpened():
    #window_handle = cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
#window_handle = cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
frameCount = 0
index=0
count=0
kernel = np.ones((5,5),np.uint8)
while cv2.getWindowProperty(WIN_NAME,0) >= 0:

    t1 = cv2.getTickCount()

    # Obtenemos un cuadro del video, y expandimos sus dimensiones a la forma
    #   [1, None, None, 3], en concordancia con lo requerido por el tensor. Una sola 
    #   columna que contiene los valores RGB de cada pixel
    ret,frame = cap.read();
    #frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)#no optimizable

    # Realizamos la deteccion de objetos, proveyendo la imagen como entrada
    (boxes, scores, classes, num) = TFSess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    #cv2.rectangle(frame,(10,10),(925,470),(0,0,255),1)
    width = frame.shape[1]
    height = frame.shape[0]
    #DEBUG
    #print(width,height)
    scores_list=np.squeeze(scores).tolist()
    while scores_list[index]>0.70:
        print(scores_list[index])
        ymin = boxes[0][index][0]*height
        xmin = boxes[0][index][1]*width
        ymax = boxes[0][index][2]*height
        xmax = boxes[0][index][3]*width
        print('ymin=',ymin,'ymax=',ymax,'xmin=',xmin)
        crop = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        
        thresh=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
     
        predicted_result = pytesseract.image_to_string(thresh) 
        image_name =str(count)+".jpg"
        cv2.imwrite(image_name,gray)
        filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
        print(filter_predicted_result)
        cv2.rectangle(frame,(int(xmin)-5,int(ymin)-35),(int(xmin)+120,int(ymin)-60),(0,0,255),-1)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,filter_predicted_result,(int(xmin)-5,int(ymin)-40),font,0.5,(255,255,255),1)
        index+=1
        count+=1
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.atleast_2d(np.squeeze(boxes)),#no optimizable
        np.atleast_1d(np.squeeze(classes).astype(np.int32)),
        np.atleast_1d(np.squeeze(scores)),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        #min_score_thresh=0.40)
        min_score_thresh=0.70)

    # Dibujamos los cuadros por segundo del video
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    # Mostramos la imagen con los dibujos superpuestos
    cv2.imshow(WIN_NAME, frame)

    # Calculo de FPS
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    frameCount+=1
    #if frameCount == 3:
    #    break

    # Al presionar Q en el teclado, finalizamos la ejecucion
    if cv2.waitKey(1) == ord('q'):
        break

#cap.stop()

cv2.destroyAllWindows()

