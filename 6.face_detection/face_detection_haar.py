# USAGE
# python face_detection_haar.py -i oriImgs/liudehua.jpg

import cv2
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, required = True, help = "path to input image")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

image = cv2.imread(args["image"])
image = imutils.resize(image, width = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 5, minSize = (30, 30), 
    flags = cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)