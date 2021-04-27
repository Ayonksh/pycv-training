import cv2
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sortContours(cnts, method = "left-to-right"):
    if method == "left-to-right":
        i = 0
    elif method == "top-to-bottom":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda x: x[1][i]))  # x[1][i]表示zip打包的元组第1个元素的第i轴
    return cnts, boundingBoxes

# 读取一个模板图像
tmpImg = cv2.imread('./oriImgs/template.png')
cv_show('tmpImg', tmpImg)

# 灰度图
tmpGray = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
cv_show('tmpGray', tmpGray)

# 二值图像
tmpBin = cv2.threshold(tmpGray, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('tmpBin', tmpBin)

# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
tmpCnts = cv2.findContours(tmpBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# 画出轮廓
cv2.drawContours(tmpImg, tmpCnts, -1, (0, 0, 255), 3)
cv_show('tmpCnts', tmpImg)

# 轮廓排序
tmpCnts = sortContours(tmpCnts)[0]

# 遍历每个轮廓
digits = {}
for (i, c) in enumerate(tmpCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(tmpImg, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv_show('tmpRect', tmpImg)
    roi = tmpBin[y:y + h, x: x + w]
    cv_show('roi', roi)
    roi = cv2.resize(roi, (55, 90))
    digits[i] = roi

# 读入待识别的图
cardImg = cv2.imread('./oriImgs/card.png')
cv_show('cardImg', cardImg)
cardGray = cv2.cvtColor(cardImg, cv2.COLOR_BGR2GRAY)
cv_show('cardGray', cardGray)

# 中值滤波，去除一些无关的噪声
cardBlur = cv2.medianBlur(cardGray, 5)
cv_show('cardBlur', cardBlur)

# 顶帽操作，突出更明亮的区域
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
cardTophat = cv2.morphologyEx(cardBlur, cv2.MORPH_TOPHAT, rectKernel)
cv_show('cardTophat', cardTophat)

# 图像梯度
gradX = cv2.Sobel(cardTophat, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = 3)
gradY = cv2.Sobel(cardTophat, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = 3)
gradX = cv2.convertScaleAbs(gradX)
gradY = cv2.convertScaleAbs(gradY)
gradDst = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 2)
res = np.hstack((gradX, gradY, gradDst))
cv_show('res', res)
cv_show('gradDst0', gradDst)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradDst = cv2.morphologyEx(gradDst, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradDist1', gradDst)

# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
gradDst = cv2.threshold(gradDst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('gradDist2', gradDst)

# 再执行闭操作（先膨胀，再腐蚀）将数字连在一起
crsKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
gradDst = cv2.morphologyEx(gradDst, cv2.MORPH_CLOSE, crsKernel)
cv_show('gradDist3', gradDst)

# 再执行闭操作
elpKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
gradDst = cv2.morphologyEx(gradDst, cv2.MORPH_CLOSE, elpKernel)
cv_show('gradDist4', gradDst)

# 画出轮廓
gradDstCnts = cv2.findContours(gradDst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cardImgCnts = cardImg.copy()
cv2.drawContours(cardImgCnts, gradDstCnts, -1, (0, 0, 255), 3)
cv_show('cardImgCnts', cardImgCnts)

# 找出要操作的轮廓并排序
gradDstCnts = gradDstCnts[5:10]  # 想要识别的卡号在第五个轮廓开始
gradDstCnts = sortContours(gradDstCnts)[0]

# 遍历每一个轮廓中的数字
output = []
for (i, c) in enumerate(gradDstCnts):
    # 初始化每组中要存放的待识别出的数组
    groupOutput = []
    # 根据坐标提取每一个组
    (gX, gY, gW, gH) = cv2.boundingRect(c)
    group = cardGray[gY - 5: gY + gH + 5, gX - 5 :gX + gW + 5]
    cv_show('group0', group)

    # 预处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group1', group)

    # 计算每一组的轮廓并排序
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    digitCnts = sortContours(digitCnts)[0]
    
    # 计算每一组中的每一个数值
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y: y + h, x: x + w]
        cv_show('roi',roi)
        roi = cv2.resize(roi, (55, 90))

        # 得分数组
        scores = []

        # 在模板中计算每一个得分
        for (digit, digitROI) in digits.items():
            res = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            score = cv2.minMaxLoc(res)[1]
            scores.append(score)

        # 返回分数数组中最大值的索引，即为匹配到的数字
        groupOutput.append(str(np.argmax(scores)))

    cv2.rectangle(cardImg, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv_show('cardImg', cardImg)
    cv2.putText(cardImg, ''.join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv_show('cardImg', cardImg)
    output.extend(groupOutput)

# 打印结果
print("Credit Card #: {}".format("".join(output)))
cv_show('cardImg', cardImg)