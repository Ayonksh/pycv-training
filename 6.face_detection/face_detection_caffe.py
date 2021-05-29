# USAGE
# python face_detection_caffe.py -i oriImgs/liudehua.jpg [-p 0.5]

import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, required = True, help = "path to input image")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe("./caffeSSD/deploy.prototxt.txt", "./caffeSSD/res10_300x300_ssd_iter_140000.caffemodel")

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > args["confidence"]:
        # 计算box坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)