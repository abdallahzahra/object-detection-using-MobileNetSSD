import numpy as np
import cv2
import os
import imutils

#all object that model can predect
AVAILABLE_CLASSES = \
["background","aeroplane","bicycle", "bird", "boat","bottle"    ,"bus","car","cat", "chair", "cow",
 "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#open cam
camera = cv2.VideoCapture(0)

#load model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

#main loop of the prog
while True:

    # read 1st frame
    colored_image = camera.read()[1]
    frame = imutils.resize(colored_image , width=800)

    # stack of object
    detected_objects = []
    
    # define hight and width of the frame 
    (h, w) = frame.shape[0:2]

    #preprocessing of the frame before  classifying "numbers are standard"
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
    
    

    net.setInput(blob)

    # forward-propagate
    detections = net.forward()
    # print(detections)
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(AVAILABLE_CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (255,0,0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    cv2.imshow("_" , frame)
    if cv2.waitKey(10) == ord('q'):
        break
camera.release()
        
    



    
