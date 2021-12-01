################################################################################
#######################方法一
#############################################################################3##
import cv2
import numpy as np
#
# cap = cv2.VideoCapture(0)
# while 1:
#     ret, frame = cap.read()
#     outline = frame.copy()  # 只有原图才能画出轮廓来
#     if ret:
#         # 将捕获的一帧图像灰度化处理
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # 边缘检测，把灰度值图像转换成二值化的图像
#         edge = cv2.Canny(gray, 50, 200, 3, L2gradient=True)
#         # 图像，轮廓像素集合，各层轮廓的索引
#         contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         #把检测到的物体画上矩形框
#         # for i in range(0, len(contours)):
#         #     x, y, w, h = cv2.boundingRect(contours[i])
#         #     cv2.rectangle(outline, (x, y), (x + w, y + h), (153, 153, 0), 2)
#         # 被画上轮廓的图像
#         cv2.drawContours(outline, contours, -1, (255,0,0), thickness=2)
#         cv2.imshow("out_win",outline)
#         if cv2.waitKey(1) & 0xff == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()

################################################################################
#######################方法二
#############################################################################3##

ball_color = 'green'

color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              'black':{'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 46])}
              }

cap = cv2.VideoCapture(0)
cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
flag = 1  # 第一次采集的照片的标志
y_init = 0 #记录第一次采集照片的纵坐标
y = 0 #记录每一次采集照片的纵坐标
dy = 0 #每一次纵坐标和第一次纵坐标的差
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)                     # 高斯模糊
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)                 # 转化成HSV图像
            erode_hsv = cv2.erode(hsv, None, iterations=2)                   # 腐蚀 粗的变细
            #将绿色以外的其他部分去掉，并将图像转换成二值化图像
            inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
            #检测二值化图像中的物体轮廓，只检测外部的轮廓
            cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            try:
                # 在检测到的轮廓中选取面积最大的那一个轮廓集合
                c = max(cnts, key=cv2.contourArea)
            except :
                print("未在图像中检测到标记物")
                continue
            #计算在上面选出的轮廓点集的最小外接矩形，这一步可以让最后画出来的矩形更好看
            rect = cv2.minAreaRect(c)
            #rect[0]是所画矩形的中心点的坐标值，1返回矩形的长宽，2返回矩形的旋转角度
            if flag:
                flag = 0
                y_init = rect[0][1]
            y = rect[0][1]
            dy = y - y_init
            print(-dy*2)
            #返回四个顶点的坐标值
            box = cv2.boxPoints(rect)
            cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)
            cv2.imshow('camera', frame)
            cv2.waitKey(1)
        else:
            print("无画面")
    else:
        print("无法读取摄像头！")

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
