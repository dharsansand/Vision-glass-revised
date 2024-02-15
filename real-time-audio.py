import numpy as np
import time
import cv2
import os
#import imutils
import subprocess
import pydub
from gtts import gTTS
from pydub import AudioSegment

pydub.AudioSegment.converter = r"C:\\yolo-master\\ffmpeg\\bin\\ffmpeg.exe"

# loading  the COCO datasets
LABELS = open("coco.names").read().strip().split(
    "\n")  # open('specify your coco names location')

# loading the yolo object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
# download yolo weights from official website and add the location here!
net = cv2.dnn.readNetFromDarknet("yolov5.cfg", "yolov5.weights")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize
cap = cv2.VideoCapture()
frame_count = 0
start = time.time()
first = True
frames = []

while True:
    frame_count += 1
# Capture frame-by-frameq
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frames.append(frame)

    if frame_count == 300:
        break
    if ret:
        key = cv2.waitKey(0)
        if frame_count % 60 == 0:
            end = time.time()

            (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []
            centers = []

            for output in layerOutputs:

                for detection in output:

                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > 0.5:

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
                        centers.append((centerX, centerY))

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            texts = []

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # find positions
                    centerX, centerY = centers[i][0], centers[i][1]

                    if centerX <= W/3:
                        W_pos = "left "
                    elif centerX <= (W/3 * 2):
                        W_pos = "center "
                    else:
                        W_pos = "right "

                    if centerY <= H/3:
                        H_pos = "top "
                    elif centerY <= (H/3 * 2):
                        H_pos = "mid "
                    else:
                        H_pos = "bottom "

                    texts.append(H_pos + W_pos + LABELS[classIDs[i]])

            print(texts)

            if texts:
                description = ', '.join(texts)
                tts = gTTS(description, lang='en')
                tts.save('tts.mp3')
                tts = pydub.AudioSegment.from_mp3("tts.mp3")
                subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])


cap.release()
cv2.destroyAllWindows()
os.remove("tts.mp3")
