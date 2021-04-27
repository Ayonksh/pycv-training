import cv2
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getVertices(box):
    rect = np.zeros((4, 2), dtype = "float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    s = box.sum(axis = 1)  # 按照y轴计算和，也就是就算坐标x和y的和
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]

    d = np.diff(box, axis = 1)  # 按照y轴计算差，也就是计算坐标x和y的差
    rect[1] = box[np.argmin(d)]
    rect[3] = box[np.argmax(d)]

    return rect

def sortContours(cnts, method = "left-to-right"):
    if method == "left-to-right":
        i = 0
    elif method == "top-to-bottom":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda x: x[1][i]))  # x[1][i]表示zip打包的元组第1个元素的第i轴
    
    return cnts, boundingBoxes

ansImg = cv2.imread('./oriImgs/answer.png')
cv_show('ans', ansImg)

ansGray = cv2.cvtColor(ansImg, cv2.COLOR_BGR2GRAY)
cv_show('ansGray', ansGray)

ansBlur = cv2.GaussianBlur(ansGray, (5, 5), 0)
cv_show('ansBlur', ansBlur)
# canny边缘检测前先用高斯滤波去掉噪声
ansEdge = cv2.Canny(ansBlur, 75, 200)
cv_show('ansEdge', ansEdge)

ansCnts = cv2.findContours(ansEdge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
ansOtl = ansImg.copy()
cv2.drawContours(ansOtl, ansCnts, -1, (0, 0, 255), 3)
cv_show('ansOtl', ansOtl)

# 根据轮廓大小进行排序
ansCnts = sorted(ansCnts, key = cv2.contourArea, reverse=True)

for c in ansCnts:
    # 计算轮廓周长
    peri = cv2.arcLength(c, True)
    # epsilon = 0.02 * peri，表示轮廓近似阈值
    # True表示封闭的
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 4个点的时候就拿出来
    if len(approx) == 4:
        docCnts = approx
        break

docCnts = docCnts.reshape(4, 2)  # 轮廓坐标点其实也是按照左上、右上、右下和左下的顺序
rect = getVertices(docCnts)  # 得到轮廓顶点

# 计算输入的w和h值
(tl, tr, br, bl) = rect

widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# 变换后对应坐标位置
dst = np.array([[0, 0], 
                [maxWidth - 1, 0], 
                [maxWidth - 1, maxHeight - 1], 
                [0, maxHeight - 1]], dtype = "float32")

# 计算变换矩阵
M = cv2.getPerspectiveTransform(rect, dst)

# 透视矫正
ansWarp = cv2.warpPerspective(ansGray, M, (maxWidth, maxHeight))
cv_show('ansWarp', ansWarp)

ansBin = cv2.threshold(ansWarp, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv_show('ansBin', ansBin)

# 找到每一个圆圈轮廓
ansCnts = cv2.findContours(ansBin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

circle1 = ansWarp.copy()
cv2.drawContours(circle1, ansCnts, -1, (0, 0, 255), 3)
# cv_show('circle1', circle1)

# 找到选项的轮廓
optionCnts = []
for c in ansCnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = w / float(h)

    # 根据实际情况指定标准
    if w >= 20 and h >= 20 and ratio >= 0.9 and ratio <= 1.1:
        optionCnts.append(c)

# 找到的轮廓按从上到下的顺序排列
optionCnts = sortContours(optionCnts, method = "top-to-bottom")[0]

circle2 = ansWarp.copy()
cv2.drawContours(circle2, optionCnts, -1, (0, 0, 255), 3)

cv2.imshow('ansBin', ansBin)
cv2.imshow('circle1', circle1)
cv2.imshow('circle2', circle2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 正确答案
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
# 答对数量
correct = 0

# 每排有5个选项，对每一排进行遍历
for (row, i) in enumerate(np.arange(0, len(optionCnts), 5)):
    cnts = sortContours(optionCnts[i: i + 5])[0]

    maxWhite = None
    # 遍历每一排的轮廓
    for (j, c) in enumerate(cnts):
        mask = np.zeros(ansBin.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)  # -1表示填充
        # cv_show('mask', mask)

        mask = cv2.bitwise_and(ansBin, ansBin, mask = mask)
        # cv_show('mask', mask)

        white = cv2.countNonZero(mask)  # 计算白点数量
        if maxWhite == None or white > maxWhite[0]:
            maxWhite = (white, j)

    # 对比答案
    color = (0, 0, 255)
    k = ANSWER_KEY[row]
    if k == maxWhite[1]:
        color = (0, 255, 0)
        correct += 1

    cv2.drawContours(ansWarp, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print('[INFO] score: {:.2f}%'.format(score))

cv2.putText(ansWarp, '{:.2f}%'.format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow('origin', ansImg)
cv2.imshow('grade', ansWarp)
cv2.imwrite('./resImgs/grade.png', ansWarp)
cv2.waitKey(0)
cv2.destroyAllWindows()