# 基于深度学习的手写数字识别
# tensorflow版本为1.*

import io
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel)
from PyQt5.Qt import (QPoint, QSize, QPixmap, QColor, QPainter, QPen, QFont)

import tensorflow as tf
from PIL import Image, ImageQt

class PaintBoard(QWidget):
    def __init__(self, Parent = None):
        super().__init__(Parent)

        self.size = QSize(420, 420)

        self.setFixedSize(self.size)

        self.board = QPixmap(self.size)
        self.board.fill(Qt.white)

        self.lastPos = QPoint(0, 0)  # 上一次鼠标位置
        self.currentPos = QPoint(0, 0)   # 当前的鼠标位置

        self.painter = QPainter()

    def clearBoard(self):
        self.board.fill(Qt.white)
        self.update()

    def getImage(self):
        # 获取画板内容（返回QImage）
        image = self.board.toImage()
        return image

    def paintEvent(self, paintEvent):
        '''
            绘图在begin()函数与end()函数间进行
            begin(param)的参数指定绘图设备，即把图画在哪里
            drawPixmap用于绘制QPixmap类型的对象
        '''
        self.painter.begin(self)
        # 0, 0为绘图的左上角起点的坐标
        self.painter.drawPixmap(0, 0, self.board)
        self.painter.end()

    def mousePressEvent(self, mouseEvent):
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.currentPos =  mouseEvent.pos()
        self.lastPos = self.currentPos

    def mouseMoveEvent(self, mouseEvent):
        # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.currentPos = mouseEvent.pos()

        self.painter.begin(self.board)
        self.painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
        self.painter.drawLine(self.lastPos, self.currentPos)
        self.painter.end()

        self.lastPos = self.currentPos

        self.update()


class Recognizer(QWidget):
    def __init__(self):
        super(Recognizer, self).__init__()

        self.paintboard = PaintBoard(self)

        self.setFixedSize(440, 500)
        self.setWindowTitle('Recognizer')

        # 新建一个垂直布局作为本窗体的主布局
        main_layout = QVBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)
        # 在主界面上方放置画板
        main_layout.addWidget(self.paintboard)

        # 新建水平子布局用于放置按键
        sub_layout = QHBoxLayout()
        # 设置此子布局和内部控件的间距为10px
        sub_layout.setContentsMargins(10, 10, 10, 10)

        # 添加一系列控件
        self.btn_recognize = QPushButton("识别", self)
        sub_layout.addWidget(self.btn_recognize)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.label_result_name = QLabel('识别结果：', self)
        sub_layout.addWidget(self.label_result_name)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        sub_layout.addWidget(self.label_result)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_clear = QPushButton("清空", self)
        sub_layout.addWidget(self.btn_clear)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("关闭", self)
        sub_layout.addWidget(self.btn_close)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

        # 将子布局加入主布局
        main_layout.addLayout(sub_layout)

    def btn_recognize_on_clicked(self):
        image = self.paintboard.getImage()
        image = ImageQt.fromqimage(image)
        image = image.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素

        recognize_result = self.recognize_img(image)  # 调用识别函数

        self.label_result.setText(str(recognize_result))  # 显示识别结果
        self.update()

    def btn_clear_on_clicked(self):
        self.paintboard.clearBoard()
        self.label_result.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()

    def recognize_img(self, img):
        img = img.convert('L')  # 转换成灰度图
        tv = list(img.getdata())  # 获取图片像素值
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]  # 转换像素范围到[0 1], 0是纯白 1是纯黑

        init = tf.global_variables_initializer()
        saver = tf.train.Saver  # 不带括号

        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.import_meta_graph('minst_cnn_model.ckpt.meta')  # 载入模型结构
            saver.restore(sess, 'minst_cnn_model.ckpt')  # 载入模型参数

            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            y_conv = graph.get_tensor_by_name("y_conv:0")

            prediction = tf.argmax(y_conv, 1)
            predint = prediction.eval(feed_dict = {x: [tva], keep_prob: 1.0}, session = sess)
            print(predint[0])
        return predint[0]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    recognizer = Recognizer()
    recognizer.show()
    sys.exit(app.exec_())