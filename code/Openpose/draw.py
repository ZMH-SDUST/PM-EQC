# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/11 20:48
@Auther ： Zzou
@File ：draw.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import cv2

# 读取图像
# image = cv2.imread('E:/HOI/pvic-main/hicodet/hico_20160224_det/images/train2015/HICO_train2015_00028831.jpg')
image = cv2.imread('E:/HOI/pvic-main/hicodet/hico_20160224_det/images/train2015/HICO_train2015_00028831.jpg')


color = (0, 255, 0)  # 矩形框颜色 (B, G, R)
thickness = 2  # 矩形框线条粗细

# 定义矩形框的左上角和右下角坐标
x1, y1 = 203, 253  # 左上角坐标
x2, y2 = 323, 514  # 右下角坐标
cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
x1, y1 = 81, 30  # 左上角坐标
x2, y2 = 330, 494  # 右下角坐标
cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# 显示图像
cv2.imshow('Image with Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
