# from PyQt5.QtWidgets import QApplication, QMessageBox,QMainWindow,  QGraphicsScene, QGraphicsPixmapItem
# from PyQt5 import uic
import time
import cv2
import numpy as np
import os
from PySide2 import QtCore, QtGui, QtWidgets  # 本程序是静态加载，把ui转成py类，创建自己的类时调用
from PySide2.QtWidgets import QApplication, QMessageBox,QGraphicsScene
from PySide2.QtUiTools import QUiLoader
from sys import argv, exit

import math
from PIL import Image, ImageStat  # pillow模块，计算亮度就靠他了
import serial  # 串口通信模块
from serial.tools import list_ports
import threading

import matplotlib.pyplot as plt
import xlwt


class Ui_MainWindow:

    def __init__(self):

        self.ui = QUiLoader().load("cameraV15.ui")

        # self.ui.button.clicked.connect(self.handleCalc)

        self.timer_camera = QtCore.QTimer()   # 定时器:界面刷新
        self.timer_light = QtCore.QTimer()   # 定时器：光源
        self.my_ser = Ser("com2", 9600)   # "COM3"
        self.flag_compute = False
        self.flag_plot = False
        self.time_start_compute = time.time()
        # self.setupUi(MainWindow)            # 界面
        # self.retranslateUi(MainWindow)      # 界面
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 准备获取图像，#选择第二个摄像头
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.CAM_NUM = 0
        # ret, frame = self.cap.read()
        # cv2.imshow('frame', frame)

        self.list_time = [0]
        self.list_value = [0]

        self.slot_init() # 设置槽函数

    def slot_init(self):
        # 设置槽函数
        self.ui.pushButton_open.clicked.connect(self.button_open_camera_click)     # 点击打开摄像头按钮，链接到打开摄像头函数
        self.timer_camera.timeout.connect(self.show_camera)        # 定时器到时槽函数（每隔一定时长刷新屏幕上的画面）
        self.timer_light.timeout.connect(self.close_light)         # 定时器timeout时启动close_light
        # self.ui.pushButton_close.clicked.connect(self.closeEvent)  # 点击关闭
        self.ui.pushButton_take.clicked.connect(self.takePhoto)    # 点击拍照
        # self.ui.pushButton_start_det.clicked.connect(self.button_start_det)
        self.ui.pushButton_start_open4C.clicked.connect(self.button_start_open4C)  # 点击设置光源

    # 实现打开摄像头功能的思路是点击‘打开摄像头’按钮后开始计时，每隔一定时长刷新屏幕上的画面，间隔很短这样在点击后就能看到捕捉的连续画面。
    def button_open_camera_click(self):
        global brightnessCount
        brightnessCount = 0

        if self.ui.spinBox_thre.value() == 0:
            QMessageBox.about(self.ui, '提示', '请输入检测阈值下限')
            return

        self.ui.label_time_2.setText(str(0))
        self.ui.label_time.setText(str(0))
        self.open_camera()   # 打开摄像头

    def start_compute(self):  # flag在打开界面时设置为true，在置为false前届面上的参数可以随时改，停止后改就没有用了
        print("start_compute")
        self.time_start_compute = time.time()
        self.flag_compute = True    # 当计算完后，置为false，停止计算时间（屏幕再怎么刷新，界面上显示的最终时间并不会变了）

    def open_camera(self):
        # 点击‘打开摄像头’按钮的槽函数//= self.cap.open(self.CAM_NUM)
        if not self.timer_camera.isActive():  # 简化if self.timer_camera.isActive() == False:
            # self.cap.open(self.CAM_NUM)
            self.timer_camera.start(100)
        else:
            self.timer_camera.start(100)
            # flag = True
            # if flag == False:
            #     QMessageBox.warning(
            #         self.ui, u"Warning", u"请检测相机与电脑是否连接正确",
            #         buttons=QMessageBox.Ok,
            #         defaultButton=QMessageBox.Ok)
            # else:
            #     self.timer_camera.start(30)

    '''
    给串口发送命令，open_camera 计算时间
    '''
    def button_start_open4C(self):
      # os.system('DPS-4C.exe')

        # 获取串口
        port_list = list(serial.tools.list_ports.comports())
        if len(port_list) <= 0:
            print('找不到串口')
        else:
            for i in range(0, len(port_list)):
                print(port_list[i])   # 打印所有串口

        send_data_open = "$1102016"     # 打开  第一通道  亮度值为100(如果命令字不是设置，而是打开，数据字没有用
        send_data_set = "$3106414"      # 设置  第一通道 亮度值为100

        self.my_ser.ser_send_data(send_data_open)  # 发送打开光源命令
        self.my_ser.ser_send_data(send_data_set)   # 发送设置光源命令
        # self.timer_light.start(5*1000)     # 5s后启动
        lightTime = self.ui.spinBox_time.value()   # 获取用户输入的打光时间N
        self.timer_light.start(lightTime * 1000)   # Ns后启动，
        # print('lightTime', self.ui.spinBox_time.value(), lightTime)

        self.open_camera()


    def close_light(self):
        send_data_setZero = "$3100016"  # 设置  第一通道亮度值为0,   使下一次打开时是不亮的
        send_data_close = "$2102015"    # 关闭  第一通道

        self.my_ser.ser_send_data(send_data_setZero)  # 发送设置光源为0命令
        self.my_ser.ser_send_data(send_data_close)  # 发送关闭光源命令

        self.start_compute()   # 开始计算时间
        self.timer_light.stop()   # 定时器关闭
        self.flag_plot = True

        self.list_value.clear()  # 清空列表数据
        self.list_time.clear()

    def plot(self, x, squares):
        plt.rcParams['font.sans-serif'] = ['SimHei']

        # 设置横坐标和纵坐标的名称
        plt.xlabel('time')
        plt.ylabel('brightness')
        # 图的标题
        plt.title('发光材料亮度图')
        # 刻度
        plt.tick_params(axis='both', labelsize=10)

        plt.scatter(x, squares, color='b', s=2, marker='.') # scatter  plot
        # plt.plot(x, squares, color='b')
        # s：点的大小/粗细 标量或array_like 默认是 rcParams['lines.markersize'] ** 2
        # c: 点的颜色   c="#ff1212",
        # marker: 标记的样式 默认是 'o',可查阅marker的类型
        plt.show()

    def saveData(self, value_list):
        now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        f = xlwt.Workbook()  # 创建工作薄
        sheet = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
        j = 0
        for i in value_list:
            sheet.write(j , 0 , i) # 循环写入，竖着写
            f.save('lightness_' + str(now_time) + '.xls')

    def text_Save(self, value_list):
        now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        file_path = 'E:\\SaveData\\'  # 新创建的txt文件的存放路径
        fileName = 'lightness_' + str(now_time)
        full_path = file_path + fileName + '.txt'
        file = open(full_path, 'w')
        # file.write(value_list)
        for i in range(len(value_list)):
            s = str(value_list[i]) + '\n'
            file.write(s)
        # file.close()

    '''
    用PIL计算亮度：平均像素，然后转换为感知亮度(采用经验公式，像素的方式处理计算)
    '''
    # def cacu_brightness(self, pic):
    #     stat = ImageStat.Stat(pic)
    #     r, g, b = stat.mean
    #     return math.sqrt(0.241 *(r ** 2)+ 0.691 *(g ** 2)+ 0.068 *(b ** 2))

    def image_brightness1(self, rgb_image):
        '''
        检测图像亮度(亮度平均值方法)
        '''
        w, h = rgb_image.size
        # 转换为HSV格式
        hsv_image = cv2.cvtColor(np.array(rgb_image, 'f'), cv2.COLOR_RGB2HSV)
        # 累计总亮度值
        sum_brightness = np.sum(hsv_image[:, :, 2])
        area = w * h  # 总像素数
        # 平均亮度
        avg = sum_brightness / area
        return avg

    def image_brightness2(self, rgb_image):
        '''
        检测图像亮度(灰度平均值方法)
        '''
        gray_image = rgb_image.convert('L')
        stat = ImageStat.Stat(gray_image)
        return stat.mean[0]

    def image_brightness3(self, rgb_image):
        '''
        检测图像亮度(基于经验公式)
        '''
        stat = ImageStat.Stat(rgb_image)
        r, g, b = stat.mean
        return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))

    def image_brightness4(self, rgb_image):
        '''
        检测图像亮度(基于RMS)
        '''
        stat = ImageStat.Stat(rgb_image)
        r, g, b = stat.rms
        return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))

    def show_camera(self):

        # ————————————————————原图, 显示在图1————————————————————
        # '''这是图1  label显示的代码'''
        # flag, self.image = self.cap.read()
        # show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
        # QtGui.QImage.Format_RGB888)  # QImage::QImage ( uchar * data, int width, int height, Format format )
        # self.ui.label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # self.ui.label_face.setScaledContents(True)

        '''这是图1  QGraphicsView显示的代码'''
        # get a frame  利用摄像头对象的read()函数读取视频的某帧
        flag, self.image = self.cap.read()
        # cv2.imshow('frame', self.image)
        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)
        # QImage::QImage ( uchar * data, int width, int height, Format format )
        scene1 = QGraphicsScene()  # 创建场景
        scene1.addPixmap(QtGui.QPixmap.fromImage(showImage))  # 给场景添加图元
        self.ui.label_face.setScene(scene1)  # 给视图窗口设置场景   # 在ui里label_face 以前是Label，后改成QGraphicsView，里面有Scene场景，可以自由缩放

        # ————————————————————灰度图, 显示在图2————————————————————
        '''这是图2  label显示的代码'''
        # gray_opencv = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # gray_image = QtGui.QImage(gray_opencv.data, show.shape[1], show.shape[0], QtGui.QImage.Format_Grayscale8)
        # self.ui.label_gray.setPixmap(QtGui.QPixmap.fromImage(gray_image))
        # self.ui.label_gray.setScaledContents(True)
        '''这是QGraphicsView显示的代码'''
        gray_opencv = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_image = QtGui.QImage(gray_opencv.data, show.shape[1], show.shape[0], QtGui.QImage.Format_Grayscale8)
        scene2 = QGraphicsScene()
        scene2.addPixmap(QtGui.QPixmap.fromImage(gray_image))
        self.ui.label_gray.setScene(scene2)

        # # ————————————————————二值化图，不显示————————————————————
        # bin_opencv = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # # binary1 = cv2.adaptiveThreshold(bin_opencv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
        # # 局部阈值（备选二值化方案）
        # ret, binary2 = cv2.threshold(bin_opencv, 0, 255,
        #                              cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)    # 全局阈值（直方图单一峰，用triangle比较好）
        # binary20 = QtGui.QImage(binary2.data, bin_opencv.shape[1], bin_opencv.shape[0], QtGui.QImage.Format_Grayscale8)
        # # self.ui.label_gray.setPixmap(QtGui.QPixmap.fromImage(binary20))
        # # self.ui.label_gray.setScaledContents(True)

        # # ————————————————————用mask提取特定区域，显示在图3————————————————————
        # mask = np.zeros(show.shape[:2], np.uint8)   # 总报错？？？？？？    shape 有三个参数，高，宽，深（位），这里只需要获取前两位
        # mask[200:280, 280:360] = 255  # 获取mask，并赋予颜色
        # gray_mask = cv2.bitwise_and(binary2, binary2, mask=mask)
        # # 通过位运算（与运算）计算带有mask的灰度图片(特别注意：只有opencv输出的图才能用opencv格式的函数，要求是矩阵；、、而QImage格式的图，作为opencv函数的输入就报错了）
        #
        # # ————用HSV设置mask————
        # # RGB转换到HSV
        # hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # # 设定绿色的阈值。确定要追踪的颜色为绿色。
        # lower_blue = np.array([26, 50, 50])
        # upper_blue = np.array([99, 255, 255])
        # # 根据阈值构建掩模，构建黑白图
        # # hsv:原图
        # # lower_blue:图像中低于这个lower_blue的值，图像值变为0,即黑色
        # # upper_blue:图像中高于这个upper_blue的值，图像值变为0
        # # 而在lower_blue～upper_blue之间的值变成255，即白色。
        # maskHSV = cv2.inRange(hsv, lower_blue, upper_blue)
        # # 对原图像和掩模进行位运算
        # # 绿色覆盖白色区域，黑色不覆盖，实现了白色转化为要追踪的绿色，也就是追踪效果。
        # res = cv2.bitwise_and(self.image, self.image, mask=maskHSV)
        # cv2.imshow('绿色', res)
        #
        # '''这是图3的label显示的代码'''
        # # gray_opencv_mask = QtGui.QImage(gray_mask.data, gray_opencv.shape[1], gray_opencv.shape[0],
        # #                                 QtGui.QImage.Format_Grayscale8)
        # # self.ui.label_gray_2.setPixmap(QtGui.QPixmap.fromImage(gray_opencv_mask))
        # # self.ui.label_gray_2.setScaledContents(True)
        #
        # '''这是图3的QGraphicsView显示的代码'''
        # gray_opencv_mask = QtGui.QImage(gray_mask.data, gray_opencv.shape[1], gray_opencv.shape[0],
        #                                 QtGui.QImage.Format_Grayscale8)
        # scene3 = QGraphicsScene()
        # scene3.addPixmap(QtGui.QPixmap.fromImage(gray_opencv_mask))
        # self.ui.label_gray_2.setScene(scene3)
        #
        # # 取roi，后续操作只对roi内的进行
        # roi = binary2[200:280, 280:360]

        # # ————计算二值化图roi部分亮点数，作为亮度（用bincount)————
        # h, w = roi.shape[:2]
        # m = np.reshape(roi, [1, w*h]).flatten()    # 将图片转化成numpy一维数组
        # m255 = np.bincount(m)                      # 用 bincount()函数
        # print("m255_Len:", len(m255))

        # # ————计算图1的roi部分亮度————
        # # 调用函数 cacu_brightness
        # roi_brightness = show[200:280, 280:360]
        # imagePIL = Image.fromarray(cv2.cvtColor(roi_brightness, cv2.COLOR_BGR2RGB))    # OpenCV转换成PIL.Image格式
        # brightness = self.cacu_brightness(imagePIL)

        # # ————计算余辉时间————
        # # 从点击拍照时开始计时，如果小于设定的采集阈值下限，并且连续小于10次，则停止计时(这是开始运行起计算，实际从点击采图按钮开始计算，点击采图开始光照并计时，再减去打光时间）
        # if len(m255) == 1:
        #     time_end = time.time()
        #     time_count = time_end - time_start
        #     self.ui.label_time_2.setText(str('%.2f' % time_count))
        #     print("m[255]=1:", m255[0])
        #     self.timer_camera.stop()
        # else:
        #     # self.ui.label_time.setText(str(m255[255]))  # 输出按亮点数算出的亮度值
        #     self.ui.label_time.setText(str('%.2f' % brightness))  # 输出PIL算出的亮度值
        #     print("m[255]=256:", m255[0], m255[255], brightness)  # 输出值为0（黑）的点数，255（白）的点数
        #     while m255[255] < self.ui.spinBox_thre.value():
        #         time_end = time.time()
        #         time_count = time_end - time_start
        #         print('totally cost', '%.2f' % time_count)
        #         # QMessageBox.about(self.ui, '余辉时间为', '%.2f' % time_count, )  # '''{time_count}'''
        #         self.ui.label_time_2.setText(str('%.2f' % time_count))
        #         self.timer_camera.stop()
        #         break

        # ~~~~~~~~~显示图3，灰度roi~~~~~~~~~~
        mask = np.zeros(show.shape[:2], np.uint8)   # shape 有三个参数，高，宽，深（位），这里只需要获取前两位
        mask[200:280, 280:360] = 255  # 获取mask，并赋予颜色
        gray_mask = cv2.bitwise_and(gray_opencv, gray_opencv, mask=mask)  # 按位与，提取mask位置，其他不显示
        gray_opencv_mask = QtGui.QImage(gray_mask.data, gray_opencv.shape[1], gray_opencv.shape[0],
                                        QtGui.QImage.Format_Grayscale8)
        scene3 = QGraphicsScene()
        scene3.addPixmap(QtGui.QPixmap.fromImage(gray_opencv_mask))
        self.ui.label_gray_2.setScene(scene3)

        # ~~~~~~~~~用5种方法分别计算roi亮度~~~~~~~~~~~
        roi_brightness = show[200:280, 280:360]
        imagePIL = Image.fromarray(cv2.cvtColor(roi_brightness, cv2.COLOR_BGR2RGB))  # OpenCV转换成PIL.Image格式
        brightness1 = self.image_brightness1(imagePIL)
        brightness2 = self.image_brightness2(imagePIL)
        brightness3 = self.image_brightness3(imagePIL)
        brightness4 = self.image_brightness4(imagePIL)

        # ~~~~~~~~~计算余辉时间~~~~~~~~~~
        self.ui.label_time.setText(str('%.2f' % brightness1))  # 输出PIL算出的亮度值
        # print("brightness:", '%.2f' % brightness1, '%.2f' % brightness2, '%.2f' % brightness3, '%.2f' % brightness4)

        # while brightness1 < self.ui.spinBox_thre.value():
        #     brightnessCount += 1
        #     if brightnessCount > 10:
        #         time_end = time.time()
        #         time_count = time_end - time_start
        #         print('totally cost', '%.2f' % time_count)
        #         # QMessageBox.about(self.ui, '余辉时间为', '%.2f' % time_count, )  # '''{time_count}'''
        #         self.ui.label_time_2.setText(str('%.2f' % time_count))
        #         self.timer_camera.stop()
        #         print('brightnessCount:',  brightnessCount)
        #         break


        time_lightoff = time.time()
        time_gap = time_lightoff - self.time_start_compute

        if self.flag_plot:
            time_x = time.time()
            list_time_x = time_x - self.time_start_compute

            self.list_time.append(list_time_x)
            self.list_value.append(brightness1)

        if self.flag_compute and brightness1 < self.ui.spinBox_thre.value() and time_gap > 5:   #  一旦第一次小于阈值，计算时间的flag为false，就不再有新的time_end，计时就相当于停止了
            time_end = time.time()
            time_count = time_end - self.time_start_compute
            print('totally cost', '%.2f' % time_count)
            # QMessageBox.about(self.ui, '余辉时间为', '%.2f' % time_count, )  # '''{time_count}'''
            self.ui.label_time_2.setText(str('%.2f' % time_count))
            self.flag_compute = False

            self.plot(self.list_time, self.list_value)   # 绘图
            print(self.list_value)
            print(self.list_time)
            # self.saveData(self.list_value)  # 存excel文件
            self.text_Save(self.list_value)  # 存txt文件

            self.list_value.clear()   # 清空列表数据
            self.list_time.clear()

    def takePhoto(self):
        if self.timer_camera.isActive():  # self.timer_camera.isActive() != False:
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            print(now_time)
            file_path = 'E:\\SaveData\\'
            cv2.imwrite(file_path + 'pic_'+str(now_time)+'.png', self.image)

            cv2.putText(self.image, 'The picture have saved !',
                        (int(self.image.shape[1]/2-130), int(self.image.shape[0]/2)),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                        1.0, (255, 0, 0), 1)

            # self.timer_camera.stop()

            # show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # 左右翻转
            #
            # # 创建场景
            # scene1 = QGraphicsScene()
            # scene2 = QGraphicsScene()
            # scene3 = QGraphicsScene()
            #
            # scene1.addPixmap(QtGui.QPixmap.fromImage(show))  # 给场景添加图元
            # self.ui.label_face.setScene(scene1)
            #
            # scene2.addPixmap(QtGui.QPixmap.fromImage(gray_image))
            # self.ui.label_gray.setScene(scene2)
            #
            # scene3.addPixmap(QtGui.QPixmap.fromImage(gray_opencv_mask))
            # self.ui.label_gray_2.setScene(scene3)

    # def closeEvent(self):
    #     if self.timer_camera.isActive():  # self.timer_camera.isActive() != False:
    #         ok = QtWidgets.QPushButton()
    #         cacel = QtWidgets.QPushButton()
    #
    #         msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")
    #
    #         msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
    #         msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
    #         ok.setText(u'确定')
    #         cacel.setText(u'取消')
    #
    #         if msg.exec_() != QtWidgets.QMessageBox.RejectRole:
    #
    #             if self.cap.isOpened():
    #                 self.cap.release()
    #             if self.timer_camera.isActive():
    #                 self.timer_camera.stop()
    #
    #             # 创建场景
    #             scene1 = QGraphicsScene()
    #             scene2 = QGraphicsScene()
    #             scene3 = QGraphicsScene()
    #             # 在场景中添加文字
    #             scene1.addText("摄像头已关闭")
    #             scene2.addText("摄像头已关闭")
    #             scene3.addText("摄像头已关闭")
    #             # 将场景加载到窗口
    #             self.ui.label_face.setScene(scene1)
    #             self.ui.label_gray.setScene(scene2)
    #             self.ui.label_gray_2.setScene(scene3)
    #
    #             # self.ui.label_face.setText("<html><head/><body><p align=\"center\"><img src=\":/newPrefix/pic/Hint.png\"/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>")
    #             # self.ui.label_gray.setText("<html><head/><body><p align=\"center\"><img src=\":/newPrefix/pic/Hint.png\"/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>")
    #             # self.ui.label_gray_2.setText("<html><head/><body><p align=\"center\"><img src=\":/newPrefix/pic/Hint.png\"/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>")

class Ser(object):
    def __init__(self, port, baud):
        self.err = 0
        # 打开串口
        try:
            self.serial = serial.Serial(port, baud, timeout=None)  # 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
            print("open serial success")
        except:
            print("open serial error!")
            self.err = -1

    # 获取串口
    def get_port_list(self):
        port_list = list(serial.tools.list_ports.comports())
        if len(port_list) == 0:
            print('找不到串口')
        else:
            for i in range(0, len(port_list)):
                print(port_list[i])
            port_serial = port_list[0]
        return  port_serial

    # 接收函数
    def ser_recv_thread(self):
        print("start ser_recv_thread")
        while(True):
            try:
                recv_data_raw = self.serial.readline()
                data = "DEVICE---->PC:" + recv_data_raw.decode()
            except:
                print("recv data error!")
                break

    # 启动接收线程
    def start_recv_thread(self):
        thread = threading.Thread(target=self.ser_recv_thread, daemon=True)
        thread.start()

    # 发送函数
    def ser_send_data(self, data):
        self.serial.write(data.encode())

    # def start_send_thread(self, data):  # ~~~~~~~~~~~~~~~~~~~~~~~~~data对不对？？？
    #     thread = threading.Thread(target= self.ser_send_data(data), daemon=True)
    #     thread.start()

    def ser_close(self):
        self.serial.close()   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~是否要关闭线程？？？


if __name__ == '__main__':
    app = QApplication(argv)

    ui_MainWindow = Ui_MainWindow()

    ui_MainWindow.ui.show()
    exit(app.exec_())
