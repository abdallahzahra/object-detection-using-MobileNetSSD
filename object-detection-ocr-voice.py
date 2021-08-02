# TODO: Abdallah Zahra
# TODO: aballahzahra9090@gmail.com


import numpy as np
import cv2
import os
import imutils
import pyttsx3
import pytesseract


#all object that model can predect
AVAILABLE_CLASSES = \
["background","aeroplane","bicycle", "bird", "boat","bottle" ,"bus","car","cat", "chair", "cow",
 "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# fun return txt
def ocr_voic(image):

    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

    img =cv2.imwrite(r"D:\PycharmProjects\finall project\out.png" ,image)
    img =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # cv2.imshow("hhh", img)v
    # cv2.waitKey(0)
    txt=pytesseract.image_to_string(img)
    print(txt)
    return txt
#open cam
camera = cv2.VideoCapture(0)
engine = pyttsx3.init()
#load model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
#main loop of the prog
while True:
    # read 1st frame
    colored_image = camera.read()[1]
    frame = imutils.resize(colored_image , width=800)
    # stack of object
    detected_objects = []
    #list of opjects
    ob_list=[]
    # define hight and width of the frame 
    (h, w) = frame.shape[0:2]
    #preprocessing of the frame before  classifying "numbers are standard"
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (224, 224), 127.5)
    net.setInput(blob)
    # forward-propagate
    detections = net.forward()
    # print(detections.shape)
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            #first dimintion 0 and then the first dim inside our first dim 0 then the dimentions which have confidence >0.7 #then select the lable element wich in this case is 1
            idx = int(detections[0, 0, i, 1])
            # print(detections[:])
            #print(detections.shape)
            #print(idx,i)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(AVAILABLE_CLASSES[idx], confidence * 100)
            ob_list.append(idx)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (255,0,0), 2)
            y = startY + 15 if startY - 15 > 15 else startY - 15 # padding
            #print(startY)
            cv2.putText(frame, label, (startX, startY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    text_say=[]
    idx_list=[]
    for i in range(len(ob_list.copy())):
        counter = ob_list[i] #[15, 15, 8]
        if counter not in idx_list:
            idx_list.append(counter)
            times=ob_list.count(counter) #2
            text_say.append(str(times)+str(AVAILABLE_CLASSES[counter])) # 2 person
    if cv2.waitKey(10) == ord('s'):
        for i in range(len(text_say)):
            engine.say(text_say[i])
            engine.runAndWait()

    if cv2.waitKey(10) == ord('v'):
        txt_ocr=ocr_voic(frame)
        print(txt_ocr)
        engine.say(txt_ocr)
        engine.runAndWait()
    cv2.imshow("_", frame)
    if cv2.waitKey(10) == ord('q'):
        break
camera.release()
