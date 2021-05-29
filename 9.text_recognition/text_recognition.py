# USAGE
# python text_recognition.py -i oriImgs/lebron_james.jpg [-p 0.05]

# 需要提前安装好Tesseract

import cv2
import numpy as np
import pytesseract
import argparse
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
ap.add_argument("-i", "--image", type = str, help = "path to input image")
ap.add_argument("-p", "--padding", type = float, default = 0.0, 
                help = "amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow('image', image)
cv2.waitKey(0)
ori = image.copy()
(oriH, oriW) = image.shape[:2]

(newW, newH) = (320, 320)
rW = oriW / float(newW)
rH = oriH / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

net = cv2.dnn.readNet('./frozen_east_text_detection.pb')
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB = True, crop = False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs = confidences)

results = []
for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # 给矩形边框填充一些，使OCR效果更好
    dX = int((endX - startX) * args["padding"])
    dY = int((endY - startY) * args["padding"])
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(oriW, endX + (dX * 2))
    endY = min(oriH, endY + (dY * 2))

    roi = ori[startY:endY, startX:endX]

    # -l 控制输入文本的语言
    # --oem（OCR 引擎模式）控制 Tesseract 使用的算法类型。1 表明我们希望仅使用深度学习 LSTM 引擎
    # --psm 控制 Tesseract 使用的自动页面分割模式，7 表明我们把 ROI 视为一行文本
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config = config)

    results.append(((startX, startY, endX, endY), text))

# 从顶到底排序
results = sorted(results, key = lambda r:r[0][1])

for ((startX, startY, endX, endY), text) in results:
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text))

    # 去除非ASCII字符，这样才能用openCV画出文本
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = ori.copy()
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Text Detection", output)
    cv2.waitKey(0)

cv2.destroyAllWindows()