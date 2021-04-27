import cv2
import numpy as np

vc = cv2.VideoCapture('./test.avi')

# 角点检测参数
featureParams = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7)  # 角点最大数量，品质因子，最短距离

# Lucas_Kanade参数
lkParams = dict(winSize = (15, 15), maxLevel = 2)

# 随机颜色条
color = np.random.randint(0, 255, (100, 3))

ret, preFrame = vc.read()
preGray = cv2.cvtColor(preFrame, cv2.COLOR_BGR2GRAY)

# 返回所有检测特征点
p0 = cv2.goodFeaturesToTrack(preGray, mask = None, **featureParams)

mask = np.zeros_like(preFrame)

while True:
    ret, curFrame = vc.read()
    if curFrame is None:
        break

    curGray = cv2.cvtColor(curFrame, cv2.COLOR_BGR2GRAY)

    p1, status, error = cv2.calcOpticalFlowPyrLK(preGray, curGray, p0, None, **lkParams)

    goodNew = p1[status == 1]
    goodOld = p0[status == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(goodNew, goodOld)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        curFrame = cv2.circle(curFrame, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(curFrame, mask)
    cv2.imshow('video', img)

    if cv2.waitKey(10) & 0xFF ==27:
        break

    # 更新每一帧
    preGray = curGray.copy()
    p0 = goodNew.reshape(-1, 1, 2)

cv2.destroyAllWindows()
vc.release()