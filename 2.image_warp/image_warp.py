import cv2
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resizeImg(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    
    return resized

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

docImg = cv2.imread('./oriImgs/page.jpg')
ratio = docImg.shape[0] / 500.0
docImg = resizeImg(docImg, height = 500)
cv_show('docImg', docImg)

docGray = cv2.cvtColor(docImg, cv2.COLOR_BGR2GRAY)
cv_show('docImg', docGray)

docBlur = cv2.GaussianBlur(docGray, (5, 5), 0)
cv_show('docBlur', docBlur)
# canny边缘检测前先用高斯滤波去掉噪声
docEdge = cv2.Canny(docBlur, 75, 200)
cv_show('docEdge', docEdge)

docCnts = cv2.findContours(docEdge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
docCnts = sorted(docCnts, key = cv2.contourArea, reverse = True)[:5]

for c in docCnts:
    # 计算轮廓周长
    peri = cv2.arcLength(c, True)
    # epsilon = 0.02 * peri，表示轮廓近似阈值
    # True表示封闭的
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 4个点的时候，说明是矩形，拿出来
    if len(approx) == 4:
        screenCnts = approx
        break

docOtl = docImg.copy()
cv2.drawContours(docOtl, [screenCnts], -1, (0, 255, 0), 2)
cv_show('docOtl', docOtl)

screenCnts = screenCnts.reshape(4, 2)  # 轮廓坐标点其实也是按照左上、右上、右下和左下的顺序
rect = getVertices(screenCnts)  # 得到轮廓顶点

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
docWarp = cv2.warpPerspective(docImg, M, (maxWidth, maxHeight))

cv2.imwrite('./resImgs/warped.jpg', docWarp)

cv2.imshow('docImg', docImg)
cv2.imshow('docOtl', docOtl)
cv2.imshow('docWarp', docWarp)
cv2.waitKey(0)
cv2.destroyAllWindows()