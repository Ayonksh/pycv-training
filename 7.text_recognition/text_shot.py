# 此项目在开源项目上进行开发
# 原项目地址：https://github.com/ianzhao05/textshot

# 使用EAST深度学习模型执行文本检测

# USAGE
# python text_shot.py [-s 1] [-p 0.05]


import io
import sys

import cv2
import numpy as np
import argparse
from imutils.object_detection import non_max_suppression

import pyperclip
import pytesseract
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

try:
    from pynotifier import Notification
except ImportError:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--scene", type = int, default = 0,
                help = "scene to recognize image. 0 as standard scene; 1 as natural scene")
ap.add_argument("-p", "--padding", type = float, default = 0.0, 
                help = "amount of padding to add to each border of ROI")
args = vars(ap.parse_args())


class Snipper(QtWidgets.QWidget):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super().__init__(parent=parent, flags=flags)

        self.setWindowTitle("TextShot")
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Dialog
        )
        self.setWindowState(self.windowState() | Qt.WindowFullScreen)

        self.screen = QtWidgets.QApplication.screenAt(QtGui.QCursor.pos()).grabWindow(0)
        palette = QtGui.QPalette()
        palette.setBrush(self.backgroundRole(), QtGui.QBrush(self.screen))
        self.setPalette(palette)

        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        self.start, self.end = QtCore.QPoint(), QtCore.QPoint()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            QtWidgets.QApplication.quit()

        return super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 100))
        painter.drawRect(0, 0, self.width(), self.height())

        if self.start == self.end:
            return super().paintEvent(event)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 3))
        painter.setBrush(painter.background())
        painter.drawRect(QtCore.QRect(self.start, self.end))
        return super().paintEvent(event)

    def mousePressEvent(self, event):
        self.start = self.end = event.pos()
        self.update()
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start == self.end:
            return super().mouseReleaseEvent(event)

        self.hide()
        QtWidgets.QApplication.processEvents()
        shot = self.screen.copy(QtCore.QRect(self.start, self.end))
        processImage(shot)
        QtWidgets.QApplication.quit()

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

def preProcess(img):
    ori = img.copy()
    (oriH, oriW) = img.shape[:2]

    (newH, newW) = (320, 320)
    rW = oriW / float(newW)
    rH = oriH / float(newH)

    img = cv2.resize(img, (newW, newH))
    (H, W) = img.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    net = cv2.dnn.readNet('./frozen_east_text_detection.pb')
    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB = True, crop = False)
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

        dX = int((endX - startX) * args["padding"])
        dY = int((endY - startY) * args["padding"])
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(oriW, endX + (dX * 2))
        endY = min(oriH, endY + (dY * 2))

        roi = ori[startY:endY, startX:endX]

        config = ("-l eng+chi_sim --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config = config)

        results.append(((startX, startY, endX, endY), text))

    results = sorted(results, key = lambda r:r[0][1])

    texts = ""
    for ((startX, startY, endX, endY), text) in results:
        texts += text + ' '

    return texts


def processImage(img):
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QBuffer.ReadWrite)
    img.save(buffer, "PNG")
    pil_img = Image.open(io.BytesIO(buffer.data()))
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    buffer.close()


    try:
        if args["scene"] == 0:
            texts = pytesseract.image_to_string(pil_img, lang = "eng+chi_sim")
        elif args["scene"] == 1:
            texts = preProcess(cv_img)
    except RuntimeError as error:
        print(f"ERROR: An error occurred when trying to process the image: {error}")
        notify(f"An error occurred when trying to process the image: {error}")
        return

    if texts:
        pyperclip.copy(texts)
        print(f'INFO: Copied "{texts}" to the clipboard')
        notify(f'Copied "{texts}" to the clipboard')
    else:
        print(f"INFO: Unable to read text from image, did not copy")
        notify(f"Unable to read text from image, did not copy")


def notify(msg):
    try:
        Notification(title="TextShot", description=msg).send()
    except (SystemError, NameError):
        trayicon = QtWidgets.QSystemTrayIcon(
            QtGui.QIcon(
                QtGui.QPixmap.fromImage(QtGui.QImage(1, 1, QtGui.QImage.Format_Mono))
            )
        )
        trayicon.show()
        trayicon.showMessage("TextShot", msg, QtWidgets.QSystemTrayIcon.NoIcon)
        trayicon.hide()


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(Qt.AA_DisableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    try:
        pytesseract.get_tesseract_version()
    except EnvironmentError:
        notify(
            "Tesseract is either not installed or cannot be reached.\n"
            "Have you installed it and added the install directory to your system path?"
        )
        print(
            "ERROR: Tesseract is either not installed or cannot be reached.\n"
            "Have you installed it and added the install directory to your system path?"
        )
        sys.exit()

    window = QtWidgets.QMainWindow()
    snipper = Snipper(window)
    snipper.show()
    sys.exit(app.exec_())
