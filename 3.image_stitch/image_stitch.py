import cv2
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def createFeature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 建立SIFT生成器
    sift = cv2.xfeatures2d.SIFT_create()
    # 检测SIFT特征点，并计算特征描述子
    (kps, features) = sift.detectAndCompute(img, None)

    # 将结果转换成NumPy数组
    kps = np.float32([kp.pt for kp in kps])

    # 返回特征点集，及对应的特征描述子
    return (kps, features)

# 读取拼接图片
imageA = cv2.imread("./oriImgs/left.png")  # 左图
imageB = cv2.imread("./oriImgs/right.png")  # 右图
cv2.imshow('imageA', imageA)
cv2.imshow('imageB', imageB)
cv2.waitKey(0)
cv2.destroyAllWindows()

#检测A、B图片的SIFT关键特征点，并计算特征描述子
(kpsA, featuresA) = createFeature(imageA)
(kpsB, featuresB) = createFeature(imageB)

# 建立暴力匹配器
matcher = cv2.BFMatcher()

# 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

goodMatches = []
for m in rawMatches:
    # 当最近距离跟次近距离的比值小于ratio(0.75)值时，保留此匹配对
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        # 存储两个点在featuresA, featuresB中的索引值
        goodMatches.append((m[0].trainIdx, m[0].queryIdx))

# 当筛选后的匹配对大于4时，才去计算视角变换矩阵
if len(goodMatches) > 4:
    # 获取匹配对的点坐标
    ptsA = np.float32([kpsA[i] for (_, i) in goodMatches])
    ptsB = np.float32([kpsB[i] for (i, _) in goodMatches])

    # 计算视角变换矩阵
    # ！！！注意A，B坐标点的顺序！！！
    (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 4.0)

    # 将图片B进行视角变换，res是变换后图片
    res = cv2.warpPerspective(imageB, H, (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
    cv_show('result', res)

    # 将图片A传入result图片最左端
    res[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
    cv_show('result', res)

    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(goodMatches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # 显示所有图片
    cv2.imshow('imageA', imageA)
    cv2.imshow('imageB', imageB)
    cv2.imshow('keypoint matches', vis)
    cv2.imwrite('./resImgs/keypointmatcher.png', vis)
    cv2.imshow('result', res)
    cv2.imwrite('./resImgs/result.png', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()