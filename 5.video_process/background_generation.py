import cv2
import numpy as np

vc = cv2.VideoCapture('./test.avi')

# 创建混合高斯模型用于背景建模
fgbg = cv2.createBackgroundSubtractorMOG2()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


while True:
    ret, frame = vc.read()
    fgmask = fgbg.apply(frame)
    # 形态学开运算去噪点
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # 寻找视频中的运动轮廓
    moveCnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 筛选行人轮廓
    for c in moveCnts:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 180:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    if cv2.waitKey(10) & 0xFF ==27:
        break
vc.release()
cv2.destroyAllWindows()