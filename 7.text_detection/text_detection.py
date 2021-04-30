# USAGE
# python text_detection.py -i oriImgs/lebron_james.jpg 

import cv2
import time
import numpy as np
import argparse
from imutils.object_detection import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow('image', image)
cv2.waitKey(0)
ori = image.copy()
(H, W) = image.shape[:2]

(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# 为EAST检测模型定义两个输出层名字：第一个是文本得分，第二个是文本形状
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
    ]

# 加载预训练EAST文本检测模型
net = cv2.dnn.readNet('./frozen_east_text_detection.pb')

# 从图像中构建一个blob，然后执行一个forward，获得两个集合
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB = True, crop = False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()
print("[INFO] text detection took {:.6f} seconds".format(end - start))

(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []
minConfidence = 0.5

for y in range(0, numRows):
    # 提取分数和包围文本的方框几何坐标
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    for x in range(0, numCols):
        if scoresData[x] < minConfidence:
            continue

        # 特征图比输入图小四倍，所以需要乘4
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

# 使用极大值抑制
boxes = non_max_suppression(np.array(rects), probs = confidences)

for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    cv2.rectangle(ori, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow('Text Detection', ori)
cv2.imwrite('./resImgs/result.jpg', ori)
cv2.waitKey(0)
cv2.destroyAllWindows()