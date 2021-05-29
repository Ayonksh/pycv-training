# 使用方法
# python text_detection_video.py [-v video.path]

import cv2
import time
import numpy as np
import argparse
import imutils
from imutils.video import FPS
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression

def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    minConfidence = 0.5

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < minConfidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help = "path to optinal input video file")
args = vars(ap.parse_args())

(W, H) = (None, None)
(newW, newH) = (320, 320)
(rW, rH) = (None, None)

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

net = cv2.dnn.readNet('./frozen_east_text_detection.pb')

if not args.get("video", False):
    print("[INFO] starting video stream...")
    vc = VideoStream(src = 0).start()
    time.sleep(1.0)

else:
    vc = cv2.VideoCapture(args["video"])

fps = FPS().start()

while True:
    frame = vc.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    frame = imutils.resize(frame, width = 1000)
    ori = frame.copy()

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)

    frame = cv2.resize(frame, (newW, newH))

    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB = True, crop = False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs = confidences)

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        cv2.rectangle(ori, (startX, startY), (endX, endY), (0, 255, 0), 2)

    fps.update()

    cv2.imshow('Text Detection', ori)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if not args.get("video", False):
    vc.stop()

else:
    vc.release()

cv2.destroyAllWindows()