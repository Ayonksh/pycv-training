# 需要安装dlib）
# http://dlib.net/files/
# pip 安装 dlib 可能会有问题，请先百度

import cv2
import numpy as np
import dlib
from collections import OrderedDict

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

faceImg = cv2.imread('./oriImgs/liudehua.jpg')
cv_show('faceImg', faceImg)

(h, w) = faceImg.shape[:2]
width = 500
r = width / float(w)
dim = (width, int(h * r))
faceImg = cv2.resize(faceImg, dim, interpolation = cv2.INTER_AREA)

faceGray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)

# 人脸检测，可能检测多个人脸
rects = detector(faceGray, 1)

for (i, rect) in enumerate(rects):
    # 对人脸框进行关键点定位
    # 转换成ndarray
    position0 = predictor(faceGray, rect)
    position = np.zeros((position0.num_parts, 2), dtype = 'int')

    for i in range(0, position0.num_parts):
        position[i] = (position0.part(i).x, position0.part(i).y)
    
    for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
        faceCopy = faceImg.copy()
        cv2.putText(faceCopy, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 根据位置画点
        for (x, y) in position[i:j]:
            cv2.circle(faceCopy, (x, y), 3, (0, 0, 255), -1)
        
        # 提取ROI区域
        (x, y, w, h) = cv2.boundingRect(np.array([position[i:j]]))
        roi = faceImg[y:y + h, x:x + w]
        (h, w) = roi.shape[:2]
        width = 250
        r = width / float(w)
        dim = (width, int(h * r))
        roi = cv2.resize(roi, dim, interpolation = cv2.INTER_AREA)

        # 显示每一部分
        cv2.imshow("ROI", roi)
        cv2.imshow("face", faceCopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

overlay = faceImg.copy()
output = faceImg.copy()

colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23), (168, 100, 168), 
          (158, 163, 32), (163, 38, 32), (180, 42, 220)]

# 遍历每个区域
for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
    # 得到每一个点的位置
    (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
    pts = position[j:k]

    # 检查位置
    if name == 'jaw':
        for l in range(1, len(pts)):
            ptA = tuple(pts[l - 1])
            ptB = tuple(pts[l])
            cv2.line(overlay, ptA, ptB, colors[i], 2)
    # 计算凸包
    else:
        hull = cv2.convexHull(pts)
        cv2.drawContours(overlay, [hull], -1, colors[i], -1)

cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

cv2.imwrite('./resImgs/output.jpg', output)
cv_show('output', output)