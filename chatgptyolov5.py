import numpy as np
import time
import cv2
import os
import imutils
import subprocess
import pydub
from gtts import gTTS
from pydub import AudioSegment

pydub.AudioSegment.converter = r"C:\\yolo-master\\ffmpeg\\bin\\ffmpeg.exe"

# loading the COCO datasets
LABELS = open("coco.names").read().strip().split(
    "\n")  # open('specify your coco names location')

# loading the YOLOv5 model
print("[INFO] loading YOLOv5 from disk...")
# download YOLOv5 weights from official website and add the location here!
net = cv2.dnn_DetectionModel("yolov3.cfg", "yolov3.weights")
net.setInputSize(640, 640)
net.setInputScale(1.0 / 255)

# initialize
cap = cv2.VideoCapture(0)
frame_count = 0
start = time.time()
first = True
frames = []

while True:
    frame_count += 1
    # Capture frame-by-frame
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
            # perform a forward pass of the YOLOv5 object detector, giving us our bounding boxes and associated probabilities
            classes, scores, boxes = net.detect(frame, 0.5, 0.5)

            # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
            centers = []
            texts = []
            for (classid, score, box) in zip(classes, scores, boxes):
                # find positions
                centerX = int((box[0] + box[2]) / 2)
                centerY = int((box[1] + box[3]) / 2)

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

                texts.append(H_pos + W_pos + LABELS[classid[0]])

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
