import uvicorn
from fastapi import File, UploadFile, FastAPI
from typing import List
from fastapi import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from PIL import Image
import base64
import io
import pickle
import argparse
import time
import io as StringIO
from io import BytesIO
import json
import os
#import torch

app = FastAPI()

confthres = 0.3
nmsthres = 0.1
yolo_path = './static'
labelsPath="configuration/obj.names"
cfgpath="configuration/yolo-obj.cfg"
wpath="weights/yolo-obj_best.weights"

def get_labels(labels_path):
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    print("[INFO] model config: ",configpath)
    print("[INFO] model weights: ", weightspath)
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='JPEG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def get_predection(image,net,LABELS,COLORS):
    print("[INFO] get_prediction() started")
    (H, W) = image.shape[:2]

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    '''
    if (torch.cuda.is_available()):
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    else:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    '''
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            print(boxes)
            print(classIDs)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return image


labelsPath="model/configuration/obj.names"
cfgpath="model/configuration/yolo-obj.cfg"
wpath="model/weights/yolo-obj_last.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)
Colors=get_colors(Lables)

templates = Jinja2Templates(directory="templates")

    
@app.get("/")
async def landing_page(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})

@app.post("/files/", status_code=201,response_class=HTMLResponse) 
async def traffic_object_recognition(request: Request,file: UploadFile):
    
    img = await file.read()
    
    img = Image.open(io.BytesIO(img))
    
    npimg=np.array(img)
    
    image=npimg.copy()
    
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    res=get_predection(image,nets,Lables,Colors)
    
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    
    np_img=Image.fromarray(image)
    img_encoded=image_to_byte_array(np_img)  
    img_bin = io.BytesIO(img_encoded)
    #return StreamingResponse(io.BytesIO(img_encoded),media_type="image/jpeg")
    return StreamingResponse(img_bin,media_type="image/jpeg")
    #return templates.TemplateResponse("index.html", {"request": request,"image":img_bin})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)
