# Author: Mr.Wang
# CreateTime: 2021/11/10
# FileName: breath_graph
# FUNCTION:
import numpy as np
import cv2
import pyqtgraph as pg
import queue
from GUI import Ui_breath
import sys
from scipy import signal
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import math
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import threading

################################################################################
#######运动历史图
################################################################################
class MHI:
    def __init__(self,frame,tau,delta,xi,t):
        self.tau=tau
        self.delta=delta
        self.xi=xi
        self.t=t
        self.data = queue.Queue()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i in range(t):
            self.data.put(frame)
        self.H = np.zeros(frame.shape)

    def getImag(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.data.put(frame)
        old_frame=self.data.get()
        a=cv2.addWeighted(old_frame.astype(float),1, frame.astype(float), -1, 0)
        D= np.fabs(a)
        Psi= D >=self.xi
        # px = np.sum(frame[Psi])
        # print(px)
        c=self.H-self.delta
        H=np.maximum(0,c)
        H[Psi]=self.tau
        p = np.sum(H)/10000
        # if p>20:
        #     p = 0
        self.H=H
        return H.astype("uint8"), p

################################################################################
#######方形标记物检测
################################################################################
class Square:
    def __init__(self):
        self.ball_color = 'green'
        self.color_dist = {'red': {'Lower': np.array([0, 60, 60]),
                              'Upper': np.array([6, 255, 255])},
                      'blue': {'Lower': np.array([100, 80, 46]),
                               'Upper': np.array([124, 255, 255])},
                      'green': {'Lower': np.array([35, 43, 35]),
                                'Upper': np.array([77, 255, 255])},
                      'black': {'Lower': np.array([0, 0, 0]),
                                'Upper': np.array([180, 255, 46])}
                      }
        # self.flag = 1  # 第一次采集的照片的标志
        # self.y_init = 0  # 记录第一次采集照片的纵坐标
        self.y = 0  # 记录每一次采集照片的纵坐标
        self.dy = 0  # 每一次纵坐标和第一次纵坐标的差
    def getrect(self,frame):
        gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
        erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
        # 将绿色以外的其他部分去掉，并将图像转换成二值化图像
        inRange_hsv = cv2.inRange(erode_hsv, self.color_dist[self.ball_color]['Lower'],
                                  self.color_dist[self.ball_color]['Upper'])
        # 检测二值化图像中的物体轮廓，只检测外部的轮廓
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        try:
            # 在检测到的轮廓中选取面积最大的那一个轮廓集合
            c = max(cnts, key=cv2.contourArea)
        except:
            print("未在图像中检测到标记物")
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return show,0
        # 计算在上面选出的轮廓点集的最小外接矩形，这一步可以让最后画出来的矩形更好看
        rect = cv2.minAreaRect(c)
        # rect[0]是所画矩形的中心点的坐标值，1返回矩形的长宽，2返回矩形的旋转角度
        # if self.flag:
        #     self.flag = 0
        #     self.y_init = rect[0][1]
        # self.y = rect[0][1]
        # self.dy = self.y - self.y_init
        self.y = rect[0][1]
        # 返回四个顶点的坐标值
        box = cv2.boxPoints(rect)
        # frame就是绘制了框的图，所以只要返回frame就可以了
        cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #因为在opencv中原点在最左上角，所以吸气是负，呼气是正数，为了和像素曲线吻合就取反
        #又因为呼吸时腹部的起伏在2-3cm左右,像素曲线在0-10，所以*2
        return frame, self.y

################################################################################
#######主程序处理：界面展示
################################################################################
class breath(QMainWindow, Ui_breath):
    def __init__(self, parent=None):
        super(breath, self).__init__(parent)
        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.CAM_NUM = 0
        self.length = 50
        self.history = np.zeros(self.length)
        self.history1 = np.zeros(self.length)
        self.i = 0
        self.flag = 1  #开始检测的标志
        self.high_flag = 0
        self.low_flag = 1
        self.y_init = 0  # 记录第一次采集照片的纵坐标
        self.rect = Square() #标记物函数追踪初始化
        self.det_start = 0 #开始检测标志
        self.pStart = 1 #运动历史图第一次获取图片
        self.color = "r"
        self.high = 0
        self.low = 0
        self.Time = 0
        self.amplitude = 2
        self.control = 0 #当满足呼吸门控的时候为1
        self.setupUi(self)
        self.start.clicked.connect(self.display)
        self.detect.clicked.connect(self.detect_set)
        self.stop.clicked.connect(self.close)
        self.timer_camera.timeout.connect(self.show_camera)


    def display(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning',
                                                    u'请检测相机与电脑是否连接正确',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                if msg==QtGui.QMessageBox.Cancel:
                    pass
            else:
                self.timer_camera.start(1)
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.original.clear()

    def show_camera(self):
        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (640, 480))
        pixel_show = show.copy()
        if self.pStart:
            self.a = MHI(show, tau=255, xi=20, delta=25, t=1)
            self.pStart = 0
        show, y = self.rect.getrect(show)
        frame, pixel = self.a.getImag(pixel_show)
        #展示标记物视频
        rectImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.original.setPixmap(QtGui.QPixmap.fromImage(rectImage))
        #展示运动历史图
        Image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                             QtGui.QImage.Format_Grayscale8)
        self.binary.setPixmap(QtGui.QPixmap.fromImage(Image))
        if self.det_start == 1 :
            if self.flag:
                self.y_init = y
                self.flag = 0
            if self.i < self.length:
                self.history[self.i] = -(y - self.y_init)
                self.detect_curve(pixel, self.history[self.i]*3, 0,0)
            self.i += 1
            if self.i > self.length:
                self.history[:-1] = self.history[1:]
                self.history[-1] = -(y - self.y_init)
                self.history1 = signal.savgol_filter(self.history,49,2)
                if pixel > 20:
                    self.y_init = y
                    self.history[-1] = 0
                    self.history1[-1] = 0
                    pixel = 20
                elif pixel < 0.2 and self.history1[-1] < self.history1[0]:
                    if self.low_flag:
                        self.low = self.history1[-1]
                        print("最低值的坐标是：", self.low,"50个之前的值是",self.history1[0])
                        self.low_flag = 0
                        self.high_flag = 1
                elif pixel < 0.2 and self.high_flag and self.history1[-1]:
                    self.high = self.history1[-1]
                    print("最高值的坐标是：",self.high,"50个之前的值是",self.history1[0])
                    self.high_flag = 0
                    self.low_flag = 1
                if self.high != 0 and self.low != 0:
                    temp = self.high - self.low
                    #第一次需要大于2，否则峰谷值就是2
                    if temp > 2:
                        self.amplitude = temp
                        print("峰谷值是：", self.amplitude)
                    else:
                        print("呼吸起伏过低，或者呼吸曲线下降，请调整呼吸")
                    if (self.high - self.history1[-1]) >= 0.8*self.amplitude:
                        self.control = 3
                    else:
                        self.control = 0
                self.detect_curve(pixel,self.history[-1]*3,self.control,self.history1[-1]*3)

    def detect_set(self):
        self.det_start = 1
        # self.a = MHI(show,tau=255, xi=20, delta=25, t=1)
        self.set_graph_ui()
        # history_detect = threading.Thread(target=self.detect_show)
        # history_detect.start()

    def set_graph_ui(self):
        pg.setConfigOptions(antialias=True)  # pyqtgraph全局变量设置函数，antialias=True开启曲线抗锯齿
        win = pg.GraphicsLayoutWidget(show=True)  # 创建widget，可实现数据界面布局自动管理
        # pg绘图窗口可以作为一个widget添加到GUI中的graph_layout，当然也可以添加到Qt其他所有的容器中
        self.graph_layout.addWidget(win)
        self.p1 = win.addPlot(title="实时呼吸曲线")  # 添加第一个绘图窗口
        self.p1.setLabel('left', text='y', color='#ffffff')  # y轴设置函数
        self.p1.showGrid(x=True, y=True)  # 栅格设置函数
        self.p1.setLogMode(x=False, y=False)  # False代表线性坐标轴，True代表对数坐标轴
        self.p1.setLabel('bottom', text='time')  # x轴设置函数
        self.y1 = np.zeros(300)
        self.curve1 = self.p1.plot(self.y1)
        self.y2 = np.zeros(300)
        self.curve2 = self.p1.plot(self.y2,pen=self.color)
        self.y3 = np.zeros(300)
        self.curve3 = self.p1.plot(self.y3, pen='y')
        self.y4 = np.zeros(300)
        self.curve4 = self.p1.plot(self.y4, pen='g')
        self.ptr = 0


    def detect_curve(self,p,dy,control,h1):
        self.y1[:-1] = self.y1[1:]
        self.y2[:-1] = self.y2[1:]
        self.y3[:-1] = self.y3[1:]
        self.y4[:-1] = self.y4[1:]
        self.y1[-1] = p
        self.y2[-1] = dy
        self.y3[-1] = control
        self.y4[-1] = h1
        self.ptr += 1
        self.curve1.setData(self.y1)
        self.curve2.setData(self.y2)
        self.curve3.setData(self.y3)
        self.curve4.setData(self.y4)
        self.curve1.setPos(self.ptr,0)
        self.curve2.setPos(self.ptr,0)
        self.curve3.setPos(self.ptr,0)
        self.curve4.setPos(self.ptr,0)


    def close(self):
        if self.cap.isOpened():
            self.cap.release()
        if self.timer_camera.isActive():
            self.timer_camera.stop()



##main 函数
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = breath()
    ui.show()
    sys.exit(app.exec_())
