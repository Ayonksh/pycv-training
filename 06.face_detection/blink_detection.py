# 需要安装dlib
# http://dlib.net/files/
# pip 安装 dlib 可能会有问题，请先百度

import cv2
import numpy as np
import dlib
from collections import OrderedDict
from scipy.spatial import distance as dist

# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def calculateEAR(eye):
    # 计算距离，竖直的
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算距离，水平的
    C = dist.euclidean(eye[0], eye[3])
    # EAR值
    EAR = (A + B) / (2.0 * C)
    return EAR

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# 加载人脸检测和关键点定位
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_models/shape_predictor_68_face_landmarks.dat')

# 分别取两个眼睛区域
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

vc = cv2.VideoCapture('./oriImgs/test.mp4')

# 设置判断参数
EAR_THRESH = 0.3  # EAR阈值
EAR_CONSEC_FRAMES = 3  # 眨眼连续帧数

# 初始化计数器
COUNTER = 0
TOTAL = 0

while True:
    ret, frame = vc.read()
    if frame is None:
        break

    (h, w) = frame.shape[:2]
    width = 1200
    r = width / float(w)
    dim = (width, int(h * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测，可能检测多个人脸
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        # 对人脸框进行关键点定位
        # 转换成ndarray
        position0 = predictor(gray, rect)
        position = np.zeros((position0.num_parts, 2), dtype = 'int')

        for i in range(0, position0.num_parts):
            position[i] = (position0.part(i).x, position0.part(i).y)

        # 分别计算EAR(eye aspect ration)值
        leftEye = position[lStart:lEnd]
        rightEye = position[rStart:rEnd]
        leftEAR = calculateEAR(leftEye)
        rightEAR = calculateEAR(rightEye)

        # 算平均值
        midEAR = (leftEAR + rightEAR) / 2.0

        # 绘制眼睛区域
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 检查是否满足阈值
        if midEAR < EAR_THRESH:
            COUNTER += 1

        else:
            # 如果连续几帧都是闭眼的，总数算一次
            if COUNTER >= EAR_CONSEC_FRAMES:
                TOTAL += 1

            # 重置
            COUNTER = 0

        # 显示
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(midEAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('video', frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break

vc.release()
cv2.destroyAllWindows()